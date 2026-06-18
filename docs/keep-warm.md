# Keeping the backend warm (avoiding Render cold starts)

Render's free tier spins the web service down after ~15 minutes of inactivity.
The next request then has to wake it, which can take **30–60 seconds** — bad for
UX and demos. This doc covers how we keep it warm.

> **The only hard fix** is upgrading Render to a paid plan (e.g. Starter, ~$7/mo),
> which never spins down. Everything below keeps a *free* instance warm, which is
> good enough for most cases but still depends on the pinger firing reliably.

## Recommended: UptimeRobot (free, reliable)

A dedicated uptime monitor pings the backend on a schedule so it never idles long
enough to sleep.

### Setup

1. Create a free account at <https://uptimerobot.com> and verify your email.
2. Dashboard → **+ New monitor**, then:

   | Field | Value |
   |---|---|
   | Monitor Type | `HTTP(s)` |
   | Friendly Name | `COMPASS backend (keep-warm)` |
   | URL | `https://compass-backend-bgej.onrender.com/health` |
   | Monitoring Interval | `5 minutes` (free-plan minimum; well under the 15-min sleep) |
   | Monitor Timeout | `60 seconds` (so a cold-start wake counts as up) |

3. (Optional) Add an email **Alert Contact** under *My Settings* so you're told
   if the backend actually goes down.

### Watch out for 503 false alarms

`/health` returns **200** only when **Redis and the model** are both up, otherwise
**503**. In production this should be 200 *if Redis is attached* (the `render.yaml`
provisions `compass-redis` and injects `REDIS_URL`) — confirm Redis is connected.

If you run without Redis on purpose (so `/health` is 503), keep-warm still works
(any request wakes the dyno), but to avoid false "down" alerts point the monitor
at the root URL instead, which always returns 200 and still wakes it:

```
https://compass-backend-bgej.onrender.com/
```

### Cost note

Keeping the service warm 24/7 means it runs continuously, consuming roughly the
full ~750 free instance-hours/month — fine for this one service, but no headroom
for a second free service.

## Backup: GitHub Action (manual)

`.github/workflows/keep-warm.yml` in the frontend repo
(`elshaddaioheha/COMPASS`) can also ping `/health`. Its **scheduled trigger is
disabled** in favour of UptimeRobot (GitHub's cron is frequently delayed past the
15-minute window, so it's unreliable for keep-warm). It can still be run manually
from the repo's **Actions** tab (Run workflow) as a backup. To re-enable the
schedule, restore the `schedule:` block in that file.

## Before a demo

Even with the above, you can guarantee a warm start by hitting the health URL a
minute or two beforehand:

```
https://compass-backend-bgej.onrender.com/health
```

The frontend also pre-warms the backend (pings `/health` on page load) and retries
cold-start responses, so the worst case is a slow first reply, not a failure.
