# Release workflow with semantic-release

This project uses [`python-semantic-release`](https://python-semantic-release.readthedocs.io/) to automate version generation while following the commit convention. The pipeline calculates the next semantic version, updates the changelog, and publishes artifacts when changes are merged into the `main` branch or when tags are created that follow the `vMAJOR.MINOR.PATCH` format.

## Commit convention

Releases are determined from commit messages using the **Conventional Commits** specification. The most common prefixes are:

- `feat:` increments the **minor** version.
- `fix:` increments the **patch** version.
- `perf:` and other improvement-compatible types also count as **patch**.
- Any commit that includes `BREAKING CHANGE:` in the body or uses the `!` suffix on the type triggers a **major** increment.

Describe changes concisely in the first line and add additional context in the commit body when necessary.

## Publishing process

1. Ensure that working branches include commits following the convention above and that automated tests pass before merging into `main`.
2. When changes reach `main`, the **Release** workflow runs `python-semantic-release` with permissions to write tags and publish to PyPI.
3. The tool determines the new version by comparing history from the latest `v*` tag, updates or creates `CHANGELOG.md`, tags the commit with the new release, and publishes the built artifacts to the configured registry.
4. If a manual release is required (for example, to rebuild an existing version), create a `vMAJOR.MINOR.PATCH` tag pointing to the desired commit and push it to the remote repository. The workflow detects the tag and repeats the publishing process.

## Troubleshooting

- If the workflow does not produce a new release, verify that the commits merged since the latest release include prefixes compatible with the convention.
- Confirm that the `PYPI_API_TOKEN` secret is configured in the repository to allow automated publishing.
- For scenarios that need pre-releases, consider using dedicated branches and adjust the semantic-release configuration to handle pre-releases.

By following these steps, new releases are managed consistently and predictably based on commit history.
