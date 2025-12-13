# DGX Spark: Connect + SSH (gx10-d56e)

## Quick Reference (from device label)
| Field | Value |
|-------|-------|
| **Hotspot SSID** | `gx10-d56e` |
| **Hotspot Password** | `e757d2df` |
| **System setup page** | `http://gx10-d56e.local` |

## 1) Connect to the Spark's Wi-Fi hotspot
1. Power on the DGX Spark and wait ~12 minutes.
2. On your laptop, join this Wi-Fi network:
   - **SSID:** `gx10-d56e`
   - **Password:** `e757d2df`

## 2) Open the setup page (optional but recommended)
Open a browser and go to:

- `http://gx10-d56e.local`

> If `.local` doesn't load:
> - Temporarily disable VPN
> - Use the Spark's IP address instead (see "Find the IP" below)

## 3) SSH into the Spark (terminal)
### Using hostname (preferred)
```bash
ssh <USERNAME>@gx10-d56e.local
```

### Using IP (if hostname fails)
```bash
ssh <USERNAME>@<SPARK_IP>
```

- Default SSH port is **22**
- On first connect, type `yes` to accept the host key, then enter your password.

## 4) Port forwarding (example)
Forward a remote service on the Spark to your laptop:

```bash
ssh -L 11000:localhost:11000 <USERNAME>@gx10-d56e.local
```

Then open `http://localhost:11000` in your browser.

## Find the Spark's IP (if needed)
### macOS
```bash
ipconfig getifaddr en0
arp -a
```

### Linux
```bash
ip route | head -n1
arp -a
```

### Windows (PowerShell / CMD)
```bat
ipconfig
```

Look for your Wi-Fi adapter's **Default Gateway** and/or the DHCP client list, then try:

- `http://<SPARK_IP>`
- `ssh <USERNAME>@<SPARK_IP>`

## Notes
- The Wi-Fi password (`e757d2df`) is **only** for joining the hotspot.
- SSH username/password is your **Linux login on the Spark** (often created during the setup page).
