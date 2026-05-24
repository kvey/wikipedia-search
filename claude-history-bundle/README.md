# Claude History Bundle

This folder contains the conversation-focused, sanitized Claude history for this project.

## Contents

- `conversations/`: JSONL conversation history and subagent metadata
- `manifest.json`: included files, truncation limits, and bundle stats

## What Changed

- Omitted Claude file-history snapshot records and operational meta records
- Kept user and assistant message turns
- Summarized tool calls and truncated large tool inputs/results
- Removed the earlier full sanitized history copy

## Stats

- Source JSONL records: 2757
- Kept records: 2667
- Skipped records: 90
- Truncated strings: 324
- Tool uses summarized: 635
- Tool results collapsed: 635
- Bundle size: 2.3M
