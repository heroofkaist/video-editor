# Security Policy

## Reporting a Vulnerability
Please report security issues privately via GitHub Security Advisories rather than opening a public issue.

## Security & Integrity Principles
- **Untrusted External Input**: Web feeds, oEmbed titles, and imported JSON metadata are treated as untrusted data and never interpolated unescaped into shell commands or FFmpeg filtergraphs.
- **No Unapproved Publishing**: Outbound social/Telegram publishing refuses execution without explicit user confirmation (`approval_confirmed=True` and non-empty `approval_reason`).
- **Workspace Path Isolation**: File serving (`/api/projects/{id}/files/...`) enforces canonical path containment inside the project workspace.
- **No Credential Leakage**: Tokens (`YOUTUBE_API_KEY`, `TELEGRAM_BOT_TOKEN`, Meta/TikTok OAuth tokens) are read strictly from environment variables and never persisted in `manifest.json` or SQLite.
