import tilelang_dsl


def main() -> None:
    package_file = getattr(tilelang_dsl, "__file__", None)
    if not package_file:
        raise SystemExit("tilelang_dsl import did not expose __file__")
    print(package_file)


if __name__ == "__main__":
    main()
