# Platform APIs, Eligibility, Quotas & Compliance Reference

Symposium Studio separates **research/discovery connectors**, **editing/rendering**, and **external publishing**. External publishing strictly requires explicit user authorization (`publishing_mode = approved_publish` + explicit approval token) and defaults to `export_only` or `draft_prep`.

## 1. Platform Publishing & Research Matrix

| Platform | Official API | Account & Auth Requirements | Rate Limits & Quotas | Media & Content Constraints | Research Access Notes |
|---|---|---|---|---|---|
| **YouTube & Shorts** | YouTube Data API v3 & Analytics API | OAuth 2.0 (`youtube.upload`, `youtube.readonly`, `yt-analytics.readonly`) | Default 10,000 units/day general pool; dedicated upload & search buckets (~100 uploads/day, ~100 searches/day). Compliance audit required to lift quota / remove private-only restriction on unverified apps. | Resumable upload via `videos.insert`; Shorts auto-classified via vertical/square aspect + <= 3 min duration; custom thumbnails via `thumbnails.set`. | Official `videos.list` / `channels.list` / `search.list` + public RSS feeds + oEmbed + explicit user-supplied URLs. Never invent CTR or retention curves. |
| **TikTok** | TikTok Content Posting API (Direct Post) & Research API | OAuth 2.0 Login Kit (`video.publish`, `user.info.basic`). Unaudited apps restricted to `SELF_ONLY` (max 5 users / 24h). | ~6 init requests/min per token; ~15 posts/day per creator. | No unauthorized superimposed third-party watermarks or forced promotional overlays. Chunked upload via `/v2/post/publish/video/init/`. | Official Research API requires academic/approved access; otherwise support explicit URL/data import & manual submission. Never bypass access controls. |
| **Instagram Reels & Posts** | Instagram Graph API | **Instagram Professional account** (Business or Creator) linked to a Facebook Page. Personal accounts unsupported. | Rolling 24h cap (typically 50–100 API-published posts/24h depending on tier). | Asynchronous 2-step container flow (`POST /{id}/media` -> `POST /{id}/media_publish`); media URL must be publicly reachable HTTPS; images must be JPEG; containers expire in 24h. | Official Business Discovery API works only for Business/Creator accounts with permitted scopes; support explicit imports/URLs. |
| **Threads** | Meta Threads API | Meta App with Threads Use Case + OAuth (`threads_basic`, `threads_content_publish`). | 250 posts / 24h, 1,000 replies / 24h per profile. | 2-step container publish (`/{id}/threads` -> `/{id}/threads_publish`). Max 500 characters per text post; max 5 links per post (`THREADS_API__LINK_LIMIT_EXCEEDED`). | Keyword/tag search & user profile endpoints require Threads API review & approved scopes; never scrape behind login walls. |
| **Telegram** | Telegram Bot API | Bot token via `@BotFather` + bot added as Channel/Group admin with post permissions. | Practical Bot API limits (~30 msgs/sec across chats; ~20 msgs/min to same group). | `sendMessage` (4096 chars), `sendPhoto`, `sendVideo` (50 MB standard Bot API limit unless using local Bot API server up to 2 GB), `sendPoll`. | Channel analytics via Telegram native stats or Bot/admin logs + tagged `?start=` / UTM referral links. |

## 2. Cross-Platform Integrity Rules

1. **No Raw Metric Equivalence**: Never compare a TikTok view directly against a YouTube view or Threads impression as identical units. Performance scores are always normalized *within* the same platform and category.
2. **No Fabricated Private Analytics**: Never invent retention rates, CTR, average view duration, or watch time when not returned by an authorized analytics API.
3. **Zero Circumvention**: Symposium Studio never solves CAPTCHAs, mimics logged-in sessions without OAuth/explicit token, or bypasses platform access restrictions.
4. **Explicit Approval Gate**: Any outbound upload or post requires explicit user confirmation (`approval_confirmed=True` and non-empty `approval_reason`).
