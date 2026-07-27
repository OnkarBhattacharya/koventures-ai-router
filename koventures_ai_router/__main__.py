"""CLI entry point for koventures_ai_router."""

from koventures_ai_router import __version__


def main() -> None:
    print(f"koventures_ai_router v{__version__}")
    print("Usage: python -m koventures_ai_router")


if __name__ == "__main__":
    main()
