---
doc_id: troubleshooting-sync
title: Troubleshooting Sync Problems and Error Codes
category: troubleshooting
tiers: [free, pro, business, enterprise]
updated: 2026-10-12
---

# Troubleshooting Sync Problems and Error Codes

## First steps for any sync problem

1. Check the AcmeCloud status page at status.acmecloud.example for an ongoing incident.
2. Make sure the desktop app is up to date (Help → Check for updates).
3. Pause sync and resume it from the tray or menu bar icon.
4. Quit and reopen the desktop app.
5. Check that the computer has at least 1 GB of free disk space.

If the problem continues after these steps, look for an error code in the app (Activity → Sync issues) and follow the matching section below.

## Error E-1042: path too long

The file path is longer than 260 characters, which Windows does not allow by default. Shorten the folder or file names, or move the file to a folder closer to the root. On Windows 10 and later, an administrator can also enable long path support in Group Policy.

## Error E-2001: storage quota exceeded

The account has used all of its storage, so new files cannot be uploaded. Existing files remain available. To fix it, delete files you no longer need and empty the Trash (files in the Trash still count toward storage), or upgrade the plan or add a storage add-on.

## Error E-3003: authentication expired

The desktop app's sign-in session has expired, usually after a password change or after 90 days without signing in. Sign out of the desktop app (Preferences → Account → Sign out) and sign back in.

## Error E-4100: file locked

Another application has the file open with an exclusive lock (common with Office files and databases). Close the file in the other application and sync resumes automatically.

## Error E-5000: server error

A temporary problem on AcmeCloud's side. Sync retries automatically every 15 minutes. If E-5000 persists for more than 1 hour and the status page shows no incident, the customer should be escalated to technical support with the time the error started.

## Large files and unsupported files

The maximum file size is 2 GB on Free and 50 GB on paid plans. Files larger than the limit are skipped. Temporary system files (such as `.tmp`, `~$` Office lock files and `.DS_Store`) are never synced.

## Selective sync

To save disk space, customers can choose which folders sync to each computer in Preferences → Sync → Selective sync. Folders that are not synced locally stay available on the web.
