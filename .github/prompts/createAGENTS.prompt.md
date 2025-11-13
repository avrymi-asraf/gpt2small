---
agent: agent
---
Generate the content for a new file named `Agents.md`. The output must strictly follow this Markdown template and schema. Populate the sections within the `[ ]` placeholders with the project's specific details.

Focus on discovering the essential knowledge that would help an AI agents be immediately productive in this codebase. Consider aspects like:
- The "big picture" architecture that requires reading multiple files to understand - major components, service boundaries, data flows, and the "why" behind structural decisions
- Critical developer workflows (builds, tests, debugging) especially commands that aren't obvious from file inspection alone
- Project-specific conventions and patterns that differ from common practices
- Integration points, external dependencies, and cross-component communication patterns

````markdown
## Project Goal

* **Description:** <Provide a 1-2 sentence description of the project's primary objective, Describe **how** the code is executed (e.g., as a command-line script, a background service, an API, Describe) **where** the code is designed to run (e.g., locally, in a specific cloud environment like AWS EMR Serverless, within a Docker container)>

---

## Project Structure

* **Architecture:** <Explain the high-level architecture, main components, and overall design. and the **Code Flow:** Describe the main code flow, detailing how data or control moves through the system from initiation to completion. You may use a numbered list or simple diagram.>

---

## File Structure

<Provide a clear, tree-like representation of the project's directory and file layout. Follow the tree with brief descriptions for the purpose of key directories and important files.>

**Example:**
\`\`\`
/project-root
├── src/
│   └── main.py     # Main application entry point
├── docs/
│   └── agents.md   # This file
├── tests/
│   └── test_main.py
└── README.md
\`\`\`

* `src/`: [Description of this directory]
* `docs/`: [Description of this directory]
* `tests/`: [Description of this directory]

---

## Building and Running

**Prerequisites:**
* <List all necessary prerequisites, dependencies (e.g., Python 3.10+, pip), and configuration steps.>

**Build Steps (if applicable):**
1.  <Step 1 for building/compiling>
2.  <Step 2...>

**Running the Application:**
1.  <Step 1 for running the code>
2.  <Step 2...>

---

## Code Writing Rules

* **Style Guide:** [Specify the code style guide to follow (e.g., PEP 8 for Python, Google C++ Style Guide).]
* **Conventions:** [List any other project-specific development rules (e.g., commit message format, branching strategy, naming conventions).]
```




- If `AGENTS.md` exists, merge intelligently - preserve valuable content while updating outdated sections
- Write concise, actionable instructions (~20-50 lines) using markdown structure
- Include specific examples from the codebase when describing patterns
- Avoid generic advice ("write tests", "handle errors") - focus on THIS project's specific approaches
- Document only discoverable patterns, not aspirational practices
- Reference key files/directories that exemplify important patterns
Source existing AI conventions from `**/{.github/copilot-instructions.md,AGENT.md,AGENTS.md,CLAUDE.md,.cursorrules,.windsurfrules,.clinerules,.cursor/rules/**,.windsurf/rules/**,.clinerules/**,README.md}` (do one glob search).

Update `AGENTS.md` for the user, then ask for feedback on any unclear or incomplete sections to iterate.