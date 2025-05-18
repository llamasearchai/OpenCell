import argparse
import json
import logging
import sys
from pathlib import Path

import networkx as nx

from ..pipeline.workflow_utils import build_workflow_graph, visualize_workflow
from .generate_workflows import create_scrna_workflow

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def create_interactive_visualization(graph: nx.DiGraph, output_path: str) -> str:
    """Create an interactive HTML visualization of the workflow"""
    try:
        import pyvis
        from pyvis.network import Network
        
        # Create network
        net = Network(
            height="800px",
            width="100%",
            bgcolor="#ffffff",
            font_color="black",
            directed=True
        )
        
        # Add nodes
        for node in graph.nodes():
            net.add_node(
                node,
                title=graph.nodes[node].get('description', ''),
                label=graph.nodes[node].get('name', node)
            )
            
        # Add edges
        for edge in graph.edges():
            net.add_edge(edge[0], edge[1])
            
        # Generate HTML
        net.show(output_path)
        return output_path
        
    except ImportError:
        logger.warning("pyvis not available - falling back to static visualization")
        return create_static_visualization(graph, output_path)

def create_static_visualization(graph: nx.DiGraph, output_path: str) -> str:
    """Create a static visualization of the workflow"""
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(graph)
    nx.draw(
        graph, 
        pos,
        with_labels=True,
        node_size=2000,
        node_color="skyblue",
        font_size=10
    )
    plt.savefig(output_path)
    return output_path

def generate_step_details_report(graph, output_dir):
    """
    Generate detailed report about each step in the workflow

    Args:
        graph: NetworkX graph of the workflow
        output_dir: Directory to save report

    Returns:
        Path to the report file
    """
    output_path = Path(output_dir) / "workflow_steps.html"

    with open(output_path, "w") as f:
        # Write HTML header
        f.write(
            """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Workflow Steps Details</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 20px; line-height: 1.6; }
                .container { max-width: 1200px; margin: 0 auto; }
                h1 { color: #333; }
                .step-card {
                    background-color: #f9f9f9;
                    border-radius: 5px;
                    padding: 20px;
                    margin-bottom: 20px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                .step-name { font-size: 1.2em; font-weight: bold; color: #333; margin-bottom: 10px; }
                .step-description { color: #666; margin-bottom: 15px; }
                .property-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
                    gap: 10px;
                }
                .property { background-color: #fff; padding: 10px; border-radius: 3px; }
                .property-name { font-weight: bold; display: inline-block; min-width: 120px; }
                pre { background-color: #f5f5f5; padding: 10px; border-radius: 3px; overflow-x: auto; }
                .COMPLETED { color: #4CAF50; }
                .RUNNING { color: #2196F3; }
                .FAILED { color: #F44336; }
                .PENDING { color: #9E9E9E; }
                .dependencies { margin-top: 10px; }
                .dependency-item { display: inline-block; background: #e3f2fd; padding: 3px 8px; border-radius: 12px; margin-right: 5px; margin-bottom: 5px; }
            </style>
        </head>
        <body>
        <div class="container">
            <h1>Workflow Steps Details</h1>
        """
        )

        # Sort nodes in topological order if possible
        try:
            nodes = list(nx.topological_sort(graph))
        except nx.NetworkXError:
            # Fall back to regular node list if not a DAG
            nodes = list(graph.nodes())

        # Write step details
        for node_id in nodes:
            node_data = graph.nodes[node_id]
            step = node_data.get("step")

            if not step:
                continue

            # Write step card
            f.write('<div class="step-card">')
            f.write(f'<div class="step-name">{step.name} (ID: {node_id})</div>')

            # Step description
            if step.description:
                f.write(f'<div class="step-description">{step.description}</div>')

            # Step properties
            f.write('<div class="property-grid">')

            # Status with color coding
            status = step.status.name if hasattr(step, "status") else "UNKNOWN"
            f.write(
                f'<div class="property"><span class="property-name">Status:</span> <span class="{status}">{status}</span></div>'
            )

            # Other properties
            f.write(f'<div class="property"><span class="property-name">Priority:</span> {step.priority}</div>')
            f.write(
                f'<div class="property"><span class="property-name">Memory:</span> {step.estimated_memory_mb} MB</div>'
            )
            f.write(
                f'<div class="property"><span class="property-name">Timeout:</span> {step.timeout_seconds or "None"} sec</div>'
            )
            f.write(f'<div class="property"><span class="property-name">Retries:</span> {step.retry_count}</div>')
            f.write(
                f'<div class="property"><span class="property-name">Uses GPU:</span> {"Yes" if step.use_gpu else "No"}</div>'
            )
            f.write(
                f'<div class="property"><span class="property-name">Separate Process:</span> {"Yes" if step.use_process else "No"}</div>'
            )

            f.write("</div>")  # Close property-grid

            # Dependencies
            if step.dependencies:
                f.write('<div class="dependencies">')
                f.write('<span class="property-name">Dependencies:</span> ')
                for dep in step.dependencies:
                    f.write(f'<span class="dependency-item">{dep}</span>')
                f.write("</div>")

            # Parameters as JSON
            if step.parameters:
                f.write('<div style="margin-top: 15px;"><span class="property-name">Parameters:</span></div>')
                f.write(f"<pre>{json.dumps(step.parameters, indent=2)}</pre>")

            f.write("</div>")  # Close step-card

        # Write HTML footer
        f.write(
            """
        </div>
        </body>
        </html>
        """
        )

    return output_path


def main():
    """Main function with enhanced options"""
    parser = argparse.ArgumentParser(
        description="Visualize workflow", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--workflow-id", help="ID of workflow to visualize (if not specified, uses scRNA-seq example)")
    parser.add_argument(
        "--format", choices=["html", "png", "pdf", "svg"], default="html", help="Output format for visualization"
    )
    parser.add_argument("--details-report", action="store_true", help="Generate detailed HTML report of steps")
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create the workflow
    if args.workflow_id:
        # Load workflow from database or storage (could be implemented)
        # For now, just use the example workflow
        logger.warning("Loading workflow by ID not implemented. Using example workflow.")
        workflow = create_scrna_workflow(workflow_id=args.workflow_id)
    else:
        workflow = create_scrna_workflow()

    # Build graph
    try:
        graph = build_workflow_graph(workflow["steps"])

        # Determine output path based on format
        if args.format == "html":
            output_path = output_dir / "workflow_visualization.html"
            viz_path = create_interactive_visualization(graph, output_path)
        else:
            output_path = output_dir / f"workflow_visualization.{args.format}"
            viz_path = create_static_visualization(graph, output_path)

        logger.info(f"Workflow visualization saved to: {viz_path}")

        # Generate detailed report if requested
        if args.details_report:
            report_path = generate_step_details_report(graph, output_dir)
            logger.info(f"Workflow details report saved to: {report_path}")

    except Exception as e:
        logger.error(f"Error creating visualization: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
