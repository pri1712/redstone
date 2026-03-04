# Contribution Guidelines

---------
### For new contributors
For contributors unfamiliar with what this repo does. Look for issues with the following tags in the [issues](https://github.com/pri1712/redstone/issues) section.
1. Documentation Improvements
2. Good first issue

You can also help me by improving test coverage or cleaning up code.

### Create a design document

In case of large architectural changes, I suggest that you create a design doc and link it to the PR and solicit comments on it before writing any code.

### Create an issue for the change

Create an issue [here](https://github.com/pri1712/redstone/issues) for any bugfixes/features that you wish to work on. Keep the issue well documented and always include what the improvemnts your fix offers.

### Create a branch for your change

```text
# 1. Fork the repository on GitHub

# 2. Clone your fork
git clone https://github.com/<your username>/redstone
cd redstone

# 3. Add upstream remote
git remote add upstream https://github.com/pri1712/tensor-cache

# 4. Build the project
cargo build

# 5. Run tests to verify everything works
cargo test
```

### Making changes

Always document changes that you do at the module level. This makes it easier for a reviewer to understand what exactly you are trying to do without having to go through all of your code.

Testing your changes

Automated tests are always recommended for contributions. Make sure you write tests so that:
1. You verify the correctness of your contribution. This serves as proof to you as well as the reviewers.
2. Make sure there are no major impacts in the case of minor bug fixes, this can be confirmed by running the benchmarks in benches/
3. Verify that your code passes integration tests.

