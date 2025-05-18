import argparse
import asyncio
import logging
import shutil  # Added for copying the logo
import signal
import sys
import time
from pathlib import Path

from ..pipeline.workflow_manager import WorkflowManager
from .generate_workflows import create_scrna_workflow

# Set up more advanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("scrna_workflow.log")],
)
logger = logging.getLogger(__name__)

# Global variable to track cancellation
cancel_requested = False


def signal_handler(sig, frame):
    """Handle Ctrl+C signal to gracefully cancel workflow"""
    global cancel_requested
    if cancel_requested:
        logger.warning("Force quitting...")
        sys.exit(1)
    else:
        logger.warning("Cancellation requested. Press Ctrl+C again to force quit.")
        cancel_requested = True


async def display_progress(workflow_manager, run_id, update_interval=1):
    """Display live progress updates for the workflow"""
    try:
        while True:
            status = await workflow_manager.get_workflow_status(run_id)

            if status["status"] in ["COMPLETED", "FAILED", "CANCELLED"]:
                break

            # Display progress bar
            progress = status["progress"]
            bar_length = 40
            filled_length = int(bar_length * progress)
            bar = "█" * filled_length + "░" * (bar_length - filled_length)

            # Get current steps
            current_steps = ", ".join([f"{s['name']} ({s['progress']:.0%})" for s in status["current_steps"]])
            if not current_steps:
                current_steps = "waiting..."

            # Print progress
            sys.stdout.write(f"\r[{bar}] {progress:.1%} | {current_steps}")
            sys.stdout.flush()

            # Check for cancellation
            global cancel_requested
            if cancel_requested:
                logger.info("Cancelling workflow...")
                await workflow_manager.cancel_workflow(run_id)
                break

            await asyncio.sleep(update_interval)
    except Exception as e:
        logger.error(f"Error in progress display: {str(e)}")
    finally:
        # Add newline after progress bar
        print()


async def main():
    """Main function with enhanced options and error handling"""
    # Set up signal handler for graceful cancellation
    signal.signal(signal.SIGINT, signal_handler)

    # Enhanced argument parser with more options
    parser = argparse.ArgumentParser(
        description="Run scRNA-seq analysis workflow", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Input/output options
    io_group = parser.add_argument_group("Input/Output Options")
    io_group.add_argument("--data-path", required=True, help="Path to data directory")
    io_group.add_argument("--output-dir", required=True, help="Directory to save results")
    io_group.add_argument("--format", choices=["10x", "h5ad", "csv", "auto"], default="auto", help="Input data format")

    # Analysis parameters
    analysis_group = parser.add_argument_group("Analysis Parameters")
    analysis_group.add_argument("--min-genes", type=int, default=200, help="Minimum genes per cell")
    analysis_group.add_argument("--min-cells", type=int, default=3, help="Minimum cells per gene")
    analysis_group.add_argument(
        "--max-pct-mito", type=float, default=20.0, help="Maximum percentage of mitochondrial genes"
    )
    analysis_group.add_argument("--n-pcs", type=int, default=50, help="Number of principal components")
    analysis_group.add_argument("--n-neighbors", type=int, default=15, help="Number of neighbors for clustering")
    analysis_group.add_argument("--resolution", type=float, default=0.8, help="Clustering resolution")
    analysis_group.add_argument(
        "--clustering-method", choices=["leiden", "louvain"], default="leiden", help="Clustering method"
    )
    analysis_group.add_argument("--batch-correction", action="store_true", help="Perform batch correction")
    analysis_group.add_argument("--batch-key", type=str, help="Column name for batch correction")

    # Execution options
    exec_group = parser.add_argument_group("Execution Options")
    exec_group.add_argument("--max-workers", type=int, default=None, help="Maximum number of parallel workers")
    exec_group.add_argument("--use-gpu", action="store_true", help="Use GPU if available")
    exec_group.add_argument("--keep-temp", action="store_true", help="Keep temporary files")
    exec_group.add_argument(
        "--execution-mode", choices=["sequential", "parallel"], default="parallel", help="Workflow execution mode"
    )
    exec_group.add_argument("--no-progress", action="store_true", help="Disable progress display")

    args = parser.parse_args()

    try:
        # Validate arguments
        data_path = Path(args.data_path)
        if not data_path.exists():
            logger.error(f"Data path does not exist: {data_path}")
            return 1

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create the workflow with enhanced parameters
        workflow = create_scrna_workflow()

        # Initialize workflow manager with execution options
        workflow_manager = WorkflowManager(
            execution_mode="SEQUENTIAL" if args.execution_mode == "sequential" else "PARALLEL",
            max_workers=args.max_workers,
        )

        # Register the workflow
        workflow_manager.register_workflow(workflow["id"], workflow["steps"])

        # Build enhanced parameters
        parameters = {
            "data_path": str(args.data_path),
            "output_dir": str(args.output_dir),
            "format_type": args.format,
            "min_genes": args.min_genes,
            "min_cells": args.min_cells,
            "max_pct_mito": args.max_pct_mito,
            "n_pcs": args.n_pcs,
            "n_neighbors": args.n_neighbors,
            "resolution": args.resolution,
            "clustering_method": args.clustering_method,
            "batch_correction": args.batch_correction,
            "batch_key": args.batch_key,
            "use_gpu": args.use_gpu,
            "keep_temp": args.keep_temp,
        }

        # Create a workflow run with parameters
        start_time = time.time()
        run_id = await workflow_manager.create_workflow_run(workflow["id"], parameters)

        logger.info(f"Created workflow run with ID: {run_id}")
        logger.info(f"Output directory: {output_dir}")

        # Start progress display in the background if enabled
        progress_task = None
        if not args.no_progress:
            progress_task = asyncio.create_task(display_progress(workflow_manager, run_id))

        # Execute the workflow
        result = await workflow_manager.execute_workflow(run_id)

        # Stop progress display
        if progress_task:
            progress_task.cancel()
            try:
                await progress_task
            except asyncio.CancelledError:
                pass

        # Calculate execution time
        execution_time = time.time() - start_time

        # Handle execution result
        if result["status"] == "COMPLETED":
            logger.info(f"Workflow completed successfully in {execution_time:.2f} seconds")
            logger.info(f"Results saved to: {result['results'].get('save_results', {}).get('output_dir')}")
        elif result["status"] == "CANCELLED":
            logger.warning(f"Workflow was cancelled after {execution_time:.2f} seconds")
        else:
            logger.error(f"Workflow failed after {execution_time:.2f} seconds: {result.get('error')}")
            for error in result.get("error_logs", []):
                logger.error(f"  {error}")
            return 1

        # Get metrics
        metrics = await workflow_manager.get_workflow_metrics(run_id)

        logger.info("Workflow metrics:")
        logger.info(f"  Total duration: {metrics['overall_metrics']['total_duration']:.2f} seconds")
        logger.info(
            f"  Steps completed: {metrics['overall_metrics']['completed_steps']}/{metrics['overall_metrics']['step_count']}"
        )

        # Create summary report if workflow completed
        if result["status"] == "COMPLETED":
            try:
                report_path = output_dir / "workflow_summary.html"
                logo_path = Path("OpenCell.svg")  # Assuming OpenCell.svg is in the root
                destination_logo_path = output_dir / "OpenCell.svg"

                # Copy logo to output directory
                if logo_path.exists():
                    shutil.copy(logo_path, destination_logo_path)
                    logo_html_tag = '<img src="OpenCell.svg" alt="OpenCell Logo" height="50" style="position:absolute; top:10px; right:10px;">'
                else:
                    logger.warning(f"Logo file {logo_path} not found. Skipping logo in report.")
                    logo_html_tag = ""

                with open(report_path, "w") as f:
                    f.write("<html><head><title>scRNA-seq Workflow Summary</title>")
                    f.write(
                        "<style>body{font-family:Arial;margin:40px;position:relative;} table{border-collapse:collapse;width:100%;margin-bottom:20px;} "
                    )  # Added position:relative for logo positioning
                    f.write("th,td{text-align:left;padding:8px;border:1px solid #ddd} ")
                    f.write("tr:nth-child(even){background-color:#f2f2f2} ")
                    f.write("th{background-color:#4CAF50;color:white}</style>")
                    f.write("</head><body>")
                    f.write(logo_html_tag)  # Add logo
                    f.write("<h1>scRNA-seq Workflow Summary</h1>")
                    f.write(
                        f"<p>Workflow Name: {metrics.get('workflow_name', 'N/A')} (Version: {metrics.get('workflow_version', 'N/A')})</p>"
                    )
                    f.write(f"<p>Run ID: {run_id}</p>")
                    f.write(f"<p>Status: {metrics.get('status', 'N/A')}</p>")
                    f.write(
                        f"<p>Execution Time: {metrics.get('duration_seconds', execution_time):.2f} seconds</p>"
                    )  # Use metrics' duration if available
                    f.write(
                        f"<p>Steps: {metrics.get('completed_steps', 0)} completed, {metrics.get('failed_steps', 0)} failed / {metrics.get('total_steps_defined', 0)} total</p>"
                    )

                    # Parameters table
                    f.write("<h2>Parameters</h2>")
                    f.write("<table><tr><th>Parameter</th><th>Value</th></tr>")
                    for key, value in parameters.items():
                        f.write(f"<tr><td>{key}</td><td>{value}</td></tr>")
                    f.write("</table>")

                    # Results table
                    f.write("<h2>Results</h2>")
                    f.write("<table><tr><th>Result</th><th>Value</th></tr>")
                    for key, value in result["results"].items():
                        if isinstance(value, dict):
                            f.write(f"<tr><td>{key}</td><td>{len(value)} items</td></tr>")
                        else:
                            f.write(f"<tr><td>{key}</td><td>{value}</td></tr>")
                    f.write("</table>")

                    # Step metrics table
                    f.write("<h2>Step Metrics</h2>")
                    f.write(
                        "<table><tr><th>Step Name</th><th>Status</th><th>Duration (s)</th><th>Retries</th><th>Error</th></tr>"
                    )
                    if "step_metrics" in metrics and metrics["step_metrics"]:
                        for step_id, step_metric_info in metrics["step_metrics"].items():
                            duration_str = (
                                f"{step_metric_info.get('duration_seconds', 0.0):.2f}"
                                if step_metric_info.get("duration_seconds") is not None
                                else "N/A"
                            )
                            error_msg = step_metric_info.get("error") or ""
                            f.write(
                                f"<tr><td>{step_metric_info.get('name', step_id)}</td><td>{step_metric_info.get('status', 'N/A')}</td>"
                            )
                            f.write(
                                f"<td>{duration_str}</td><td>{step_metric_info.get('retry_attempts', 0)}</td><td>{error_msg}</td></tr>"
                            )
                    else:
                        f.write("<tr><td colspan='5'>No step metrics available.</td></tr>")
                    f.write("</table>")

                    f.write("</body></html>")

                logger.info(f"Summary report created: {report_path}")
            except Exception as e:
                logger.error(f"Failed to create summary report: {str(e)}")

        return 0

    except KeyboardInterrupt:
        logger.warning("Execution interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Error executing workflow: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    # Run with proper exit code
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
