"""CLI entry point for the data ingestion pipeline.

Usage:
  # Single repository
  python scripts/ingest.py --repo https://github.com/owner/repo

  # Multiple repositories
  python scripts/ingest.py --repo https://github.com/a/one --repo https://github.com/b/two

  # All repositories from REPO_URLS config
  python scripts/ingest.py --all-repos
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from codebase_rag.config import Config
from codebase_rag.data_ingestion.pipeline import IngestPipeline


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Ingest data from the repository into the vector database")

    parser.add_argument("--dirs", nargs="+", help="List of directories to include (default: docs, src, tests)")

    parser.add_argument(
        "--files", nargs="+", help="List of specific files to include (default: README.md, pyproject.toml)"
    )

    parser.add_argument(
        "--repo",
        type=str,
        action="append",
        dest="repos",
        help="GitHub repository URL to ingest (can be specified multiple times)",
    )

    parser.add_argument(
        "--all-repos",
        action="store_true",
        help="Ingest all repositories from REPO_URLS config variable",
    )

    parser.add_argument(
        "--drop",
        action="store_true",
        help="Drop existing collections before ingestion (alias for --force)",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force full re-index by dropping existing collection first",
    )

    parser.add_argument("--no-cache", action="store_true", help="Don't use document cache")

    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    return parser.parse_args()


def main() -> None:
    """Main entry point for the script."""
    args = parse_arguments()

    if not args.repos and not args.all_repos:
        sys.stderr.write("Error: one of --repo or --all-repos is required.\n")
        sys.exit(1)

    # Determine repos to ingest
    repo_urls: list[str] | None = args.repos
    if args.all_repos:
        config = Config.get_instance()
        repo_urls = list(config.repo_urls)

    # Create the ingestion pipeline
    pipeline = IngestPipeline(
        included_dirs=args.dirs,
        included_files=args.files,
        drop_existing=args.drop or args.force,
        use_cache=not args.no_cache,
        debug=args.debug,
        repo_urls=repo_urls,
    )

    # Run the pipeline
    pipeline.run()


if __name__ == "__main__":
    main()
