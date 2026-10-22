"""CLI entry point for accolaide_ai_router."""

from accolaide_ai_router import __version__


def main() -> None:
    print(f"accolaide_ai_router v{__version__}")
    print("Usage: python -m accolaide_ai_router")


if __name__ == "__main__":
    main()
