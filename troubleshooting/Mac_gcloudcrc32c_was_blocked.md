# Troubleshooting **gcloud‑crc32c** & “Source‑hash does not match destination‑hash” errors on macOS

> **Scenario**  When you try to copy data from Google Cloud Storage with the `gcloud` CLI on macOS, you may hit **two different errors** that are actually the *same* root cause.
>
> 1. **Gatekeeper alert**
>    `"gcloud-crc32c" can’t be opened because Apple cannot check it for malicious software.`
> 2. **gcloud copy failure**
>    `ERROR: Source hash q7f3+A== does not match destination hash AAAAAA== for object …`

---

## 1  Why do these happen?

|                | Gatekeeper popup                                                       | Hash‑mismatch error                                                                   |
| -------------- | ---------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| **Trigger**    | First time `gcloud` tries to run its helper binary **`gcloud-crc32c`** | `gcloud storage cp` compares the server‑side CRC‑32C to a client‑side CRC‑32C         |
| **Root cause** | The binary isn’t *notarised* → macOS *quarantines* it                  | Because the helper is blocked, the client CRC is never computed → `AAAAAA==` sentinel |
| **Effect**     | macOS blocks execution & shows a warning                               | The CLI detects mismatched CRCs and aborts the copy                                   |

<details>
<summary>What is <code>gcloud-crc32c</code>?</summary>

A tiny helper executable that computes CRC‑32C hashes. The Storage API uses CRC‑32C to ensure every uploaded/downloaded object is intact. When the helper is blocked, the CLI cannot verify file integrity.

</details>

---

## 2  Quick fixes (pick one)

### A. “Open Anyway” in System Settings

1. **Apple menu ▸ System Settings ▸ Privacy & Security**.
2. At the bottom you’ll see *“gcloud-crc32c was blocked.”*
   Click **Allow Anyway** → run your command again.

### B. Remove the quarantine attribute via Terminal

```bash
# Change the Homebrew prefix if yours is not /opt/homebrew or /usr/local
xattr -d com.apple.quarantine \
  "$(brew --prefix)/Caskroom/google-cloud-sdk/latest/google-cloud-sdk/bin/gcloud-crc32c"
```

Run your `download_data.sh` (or any `gcloud storage cp`) again – it should succeed.

---

## 3  Permanent / team‑friendly solutions

| Option                                        | Command                                                                                     | Pros                                                        | Cons                                                      |
| --------------------------------------------- | ------------------------------------------------------------------------------------------- | ----------------------------------------------------------- | --------------------------------------------------------- |
| **Upgrade Google Cloud SDK**                  | `brew upgrade --cask google-cloud-sdk` <br>or `gcloud components update`                    | Newer builds are signed & notarised → no pop‑ups for anyone | Requires everyone to upgrade                              |
| **Disable hash checking** *(not recommended)* | `gcloud config set storage/check_hashes never` <br>or add `--no-check-hash` to your command | Works even if helper is blocked                             | Skips integrity validation on every copy                  |
| **Ship a setup script**                       | Add the `xattr -d …` line to your project’s setup script                                    | Keeps hash checks; works offline                            | Still executes unsigned code – users must trust your repo |

---

## 4  Why your script is not at fault

Your `download_data.sh` simply wraps:

```bash
gcloud storage cp "gs://<BUCKET>/<OBJECT>" "${DESTINATION}"
```

The failure is entirely local – once macOS allows `gcloud-crc32c`, the same script finishes without changes.

---

## 5  Reference commands

```bash
# Verify the helper binary exists
ls -l "$(brew --prefix)/Caskroom/google-cloud-sdk/latest/google-cloud-sdk/bin/gcloud-crc32c"

# Check for the quarantine attribute
xattr "$(brew --prefix)/Caskroom/google-cloud-sdk/latest/google-cloud-sdk/bin/gcloud-crc32c"

# Remove quarantine (if present)
xattr -d com.apple.quarantine "$(brew --prefix)/Caskroom/google-cloud-sdk/latest/google-cloud-sdk/bin/gcloud-crc32c"
```

---

### Credits & last tested

* macOS 14.4 Sonoma
* Google Cloud SDK 475.0.0 (Homebrew Cask)
* Homebrew 4.3.2

Feel free to open an issue or PR if you find newer SDK versions still affected.
