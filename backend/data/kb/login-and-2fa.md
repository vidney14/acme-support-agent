---
doc_id: login-and-2fa
title: Sign-in, Passwords and Two-Factor Authentication
category: security
tiers: [free, pro, business, enterprise]
updated: 2026-06-30
---

# Sign-in, Passwords and Two-Factor Authentication

## Resetting a password

Use "Forgot password" on the sign-in page. AcmeCloud emails a reset link that is valid for 30 minutes. Support agents never ask for, and must never accept, a customer's password.

## Account lockout

After 5 failed sign-in attempts in a row, the account is locked for 30 minutes. The lock clears automatically. Support cannot unlock an account early, because the lock protects against password-guessing attacks.

## Two-factor authentication (2FA)

2FA can use an authenticator app or a hardware security key. When 2FA is turned on, AcmeCloud shows 10 one-time recovery codes. Each code can be used once.

### Lost 2FA device

If the customer lost their phone or security key, they can sign in with one of their recovery codes and then set up 2FA again on a new device.

If the customer has no recovery codes, the account must go through manual identity verification by the security team. This takes 2 to 3 business days. The AI assistant cannot remove 2FA from an account. It must create a P1 escalation ticket in the security category.

## Single sign-on (SSO)

On Business and Enterprise accounts that use SAML SSO, sign-in is controlled by the customer's own identity provider (for example Okta or Microsoft Entra ID). If an SSO user cannot sign in, they must contact their company's IT administrator. AcmeCloud support cannot bypass SSO.

## Suspected account compromise

If a customer believes someone else has accessed their account, they should reset their password immediately and sign out all sessions (Settings → Security → Sign out everywhere). The assistant must create a P1 escalation ticket in the security category so the security team can review recent activity.
