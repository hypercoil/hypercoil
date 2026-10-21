# hypercoil: agent guide

hypercoil is a member of the diffprog federation. Its role, boundaries and neighbours are in
`diffprog-dev/context/repos/hypercoil.md`. The federation's rules are in `diffprog-dev/AGENTS.md`;
read it first when diffprog-dev is checked out next to this repo.

## Rules

- Read remote and published resources; never write to them. Pushing, pull requests,
  releases and uploads are left to the user.
- Run the tests with `pytest` in the federation venv. Mark a test that needs a GPU, a
  licence or an optional library with `@pytest.mark.requires(<capability>)`.
