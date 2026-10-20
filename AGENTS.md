# Repository exploration

This is a large research repository.

When working on a task:

1. Do not scan or analyze the entire repository by default.
2. Start from the files or module most directly related to the task.
3. Follow imports, callers, dependencies, data flow, and configuration only when
   they are relevant to understanding the problem.
4. Expand the search scope when there is a concrete reason to believe the issue
   may originate outside the current module.
5. Do not artificially restrict investigation if the root cause may be elsewhere.
6. Prefer targeted searches for symbols, usages, imports, and call sites over
   reading whole directories.
7. Avoid unrelated modules, generated files, datasets, recordings, logs,
   build artifacts, caches, and experimental outputs unless the task requires them.
8. For algorithmic/research tasks, prioritize correctness and discovering the
   real cause over minimizing exploration. Explore additional modules when this
   can materially affect the conclusion.