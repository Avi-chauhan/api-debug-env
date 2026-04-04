"""
30 API spec templates covering 6 real-world domains.

Each spec defines:
- api_name: Human-readable name
- http_method: GET, POST, PUT, PATCH, DELETE
- endpoint: API path
- required_headers: Headers that must be present
- required_fields: Fields that must be in the request body
- optional_fields: Fields that may be in the request body
- field_types: Expected type for each field (used for validation)
- valid_example: A correct request body (used to generate broken requests)

Supported field types:
- "string", "integer", "float", "boolean"
- "email" (validated by regex)
- "datetime" (ISO 8601 format)
- "enum:val1,val2,val3" (one of the listed values)
- "url" (validated by pattern)
- "phone" (validated by pattern)
- "object" (nested dict, not deeply validated)
- "array" (list)
"""

from typing import Any, Dict, List


def _spec(
    api_name: str,
    http_method: str,
    endpoint: str,
    required_fields: List[str],
    field_types: Dict[str, str],
    valid_example: Dict[str, Any],
    optional_fields: List[str] = None,
    required_headers: Dict[str, str] = None,
) -> Dict[str, Any]:
    """Build a spec dict with sensible defaults."""
    return {
        "api_name": api_name,
        "http_method": http_method,
        "endpoint": endpoint,
        "required_headers": required_headers or {
            "Authorization": "Bearer sk_test_abc123",
            "Content-Type": "application/json",
        },
        "required_fields": required_fields,
        "optional_fields": optional_fields or [],
        "field_types": field_types,
        "valid_example": valid_example,
    }


# =========================================================================
# Domain 1: Payment APIs (Stripe-like)
# =========================================================================

PAYMENT_SPECS = [
    _spec(
        api_name="Create Customer",
        http_method="POST",
        endpoint="/v1/customers",
        required_fields=["email", "name"],
        optional_fields=["phone", "description", "address"],
        field_types={
            "email": "email",
            "name": "string",
            "phone": "phone",
            "description": "string",
            "address": "object",
        },
        valid_example={
            "email": "alice@example.com",
            "name": "Alice Johnson",
        },
    ),
    _spec(
        api_name="Create Charge",
        http_method="POST",
        endpoint="/v1/charges",
        required_fields=["amount", "currency", "customer_id"],
        optional_fields=["description", "receipt_email"],
        field_types={
            "amount": "integer",
            "currency": "enum:usd,eur,gbp,inr,jpy",
            "customer_id": "string",
            "description": "string",
            "receipt_email": "email",
        },
        valid_example={
            "amount": 2500,
            "currency": "usd",
            "customer_id": "cus_abc123",
        },
    ),
    _spec(
        api_name="Create Subscription",
        http_method="POST",
        endpoint="/v1/subscriptions",
        required_fields=["customer_id", "plan_id", "start_date"],
        optional_fields=["trial_days", "auto_renew"],
        field_types={
            "customer_id": "string",
            "plan_id": "string",
            "start_date": "datetime",
            "trial_days": "integer",
            "auto_renew": "boolean",
        },
        valid_example={
            "customer_id": "cus_abc123",
            "plan_id": "plan_monthly_pro",
            "start_date": "2026-04-01T00:00:00Z",
        },
    ),
    _spec(
        api_name="Process Refund",
        http_method="POST",
        endpoint="/v1/refunds",
        required_fields=["charge_id", "amount"],
        optional_fields=["reason"],
        field_types={
            "charge_id": "string",
            "amount": "integer",
            "reason": "enum:duplicate,fraudulent,requested_by_customer",
        },
        valid_example={
            "charge_id": "ch_abc123",
            "amount": 1500,
        },
    ),
    _spec(
        api_name="List Transactions",
        http_method="GET",
        endpoint="/v1/transactions",
        required_fields=["account_id"],
        optional_fields=["start_date", "end_date", "limit"],
        field_types={
            "account_id": "string",
            "start_date": "datetime",
            "end_date": "datetime",
            "limit": "integer",
        },
        valid_example={
            "account_id": "acc_abc123",
        },
    ),
]

# =========================================================================
# Domain 2: User Management
# =========================================================================

USER_SPECS = [
    _spec(
        api_name="Create User",
        http_method="POST",
        endpoint="/api/users",
        required_fields=["email", "username", "password"],
        optional_fields=["full_name", "role"],
        field_types={
            "email": "email",
            "username": "string",
            "password": "string",
            "full_name": "string",
            "role": "enum:admin,editor,viewer",
        },
        valid_example={
            "email": "bob@example.com",
            "username": "bob_smith",
            "password": "SecurePass123!",
        },
    ),
    _spec(
        api_name="Update Profile",
        http_method="PATCH",
        endpoint="/api/users/{user_id}/profile",
        required_fields=["user_id", "display_name"],
        optional_fields=["bio", "avatar_url", "timezone"],
        field_types={
            "user_id": "string",
            "display_name": "string",
            "bio": "string",
            "avatar_url": "url",
            "timezone": "string",
        },
        valid_example={
            "user_id": "usr_abc123",
            "display_name": "Bob Smith",
        },
    ),
    _spec(
        api_name="Reset Password",
        http_method="POST",
        endpoint="/api/auth/reset-password",
        required_fields=["email"],
        optional_fields=["redirect_url"],
        field_types={
            "email": "email",
            "redirect_url": "url",
        },
        valid_example={
            "email": "bob@example.com",
        },
    ),
    _spec(
        api_name="Verify Email",
        http_method="POST",
        endpoint="/api/auth/verify-email",
        required_fields=["token", "email"],
        field_types={
            "token": "string",
            "email": "email",
        },
        valid_example={
            "token": "verify_abc123xyz",
            "email": "bob@example.com",
        },
    ),
    _spec(
        api_name="Delete Account",
        http_method="DELETE",
        endpoint="/api/users/{user_id}",
        required_fields=["user_id", "confirmation"],
        field_types={
            "user_id": "string",
            "confirmation": "enum:DELETE,CONFIRM",
        },
        valid_example={
            "user_id": "usr_abc123",
            "confirmation": "DELETE",
        },
    ),
]

# =========================================================================
# Domain 3: Content APIs (GitHub-like)
# =========================================================================

CONTENT_SPECS = [
    _spec(
        api_name="Create Repository",
        http_method="POST",
        endpoint="/api/repos",
        required_fields=["name", "visibility"],
        optional_fields=["description", "auto_init", "license"],
        field_types={
            "name": "string",
            "visibility": "enum:public,private,internal",
            "description": "string",
            "auto_init": "boolean",
            "license": "string",
        },
        valid_example={
            "name": "my-project",
            "visibility": "public",
        },
    ),
    _spec(
        api_name="Create Issue",
        http_method="POST",
        endpoint="/api/repos/{repo_id}/issues",
        required_fields=["title", "repo_id"],
        optional_fields=["body", "assignee", "labels", "priority"],
        field_types={
            "title": "string",
            "repo_id": "string",
            "body": "string",
            "assignee": "string",
            "labels": "array",
            "priority": "enum:low,medium,high,critical",
        },
        valid_example={
            "title": "Fix login page redirect",
            "repo_id": "repo_abc123",
        },
    ),
    _spec(
        api_name="Create Comment",
        http_method="POST",
        endpoint="/api/issues/{issue_id}/comments",
        required_fields=["issue_id", "body"],
        optional_fields=["mentions"],
        field_types={
            "issue_id": "string",
            "body": "string",
            "mentions": "array",
        },
        valid_example={
            "issue_id": "issue_abc123",
            "body": "This looks like a duplicate of #42.",
        },
    ),
    _spec(
        api_name="Merge Pull Request",
        http_method="PUT",
        endpoint="/api/repos/{repo_id}/pulls/{pr_id}/merge",
        required_fields=["repo_id", "pr_id", "merge_method"],
        optional_fields=["commit_title", "delete_branch"],
        field_types={
            "repo_id": "string",
            "pr_id": "string",
            "merge_method": "enum:merge,squash,rebase",
            "commit_title": "string",
            "delete_branch": "boolean",
        },
        valid_example={
            "repo_id": "repo_abc123",
            "pr_id": "pr_456",
            "merge_method": "squash",
        },
    ),
    _spec(
        api_name="Create Release",
        http_method="POST",
        endpoint="/api/repos/{repo_id}/releases",
        required_fields=["repo_id", "tag_name", "name"],
        optional_fields=["body", "draft", "prerelease"],
        field_types={
            "repo_id": "string",
            "tag_name": "string",
            "name": "string",
            "body": "string",
            "draft": "boolean",
            "prerelease": "boolean",
        },
        valid_example={
            "repo_id": "repo_abc123",
            "tag_name": "v1.0.0",
            "name": "Version 1.0.0",
        },
    ),
]

# =========================================================================
# Domain 4: Messaging (Twilio-like)
# =========================================================================

MESSAGING_SPECS = [
    _spec(
        api_name="Send SMS",
        http_method="POST",
        endpoint="/api/messages/sms",
        required_fields=["to", "from_number", "body"],
        optional_fields=["callback_url"],
        field_types={
            "to": "phone",
            "from_number": "phone",
            "body": "string",
            "callback_url": "url",
        },
        valid_example={
            "to": "+14155551234",
            "from_number": "+14155550000",
            "body": "Your verification code is 123456",
        },
    ),
    _spec(
        api_name="Send Email",
        http_method="POST",
        endpoint="/api/messages/email",
        required_fields=["to_email", "subject", "body"],
        optional_fields=["cc", "bcc", "reply_to"],
        field_types={
            "to_email": "email",
            "subject": "string",
            "body": "string",
            "cc": "email",
            "bcc": "email",
            "reply_to": "email",
        },
        valid_example={
            "to_email": "customer@example.com",
            "subject": "Order Confirmation",
            "body": "Your order #1234 has been confirmed.",
        },
    ),
    _spec(
        api_name="Create Webhook",
        http_method="POST",
        endpoint="/api/webhooks",
        required_fields=["url", "events"],
        optional_fields=["secret", "active"],
        field_types={
            "url": "url",
            "events": "array",
            "secret": "string",
            "active": "boolean",
        },
        valid_example={
            "url": "https://myapp.com/webhook",
            "events": ["message.sent", "message.delivered"],
        },
    ),
    _spec(
        api_name="Create Template",
        http_method="POST",
        endpoint="/api/templates",
        required_fields=["name", "content", "channel"],
        optional_fields=["variables", "language"],
        field_types={
            "name": "string",
            "content": "string",
            "channel": "enum:sms,email,push",
            "variables": "array",
            "language": "string",
        },
        valid_example={
            "name": "welcome_message",
            "content": "Hello {{name}}, welcome to our service!",
            "channel": "email",
        },
    ),
    _spec(
        api_name="Verify Phone",
        http_method="POST",
        endpoint="/api/verify/phone",
        required_fields=["phone_number", "code"],
        field_types={
            "phone_number": "phone",
            "code": "string",
        },
        valid_example={
            "phone_number": "+14155551234",
            "code": "123456",
        },
    ),
]

# =========================================================================
# Domain 5: E-Commerce
# =========================================================================

ECOMMERCE_SPECS = [
    _spec(
        api_name="Create Order",
        http_method="POST",
        endpoint="/api/orders",
        required_fields=["customer_id", "items", "shipping_address"],
        optional_fields=["notes", "coupon_code"],
        field_types={
            "customer_id": "string",
            "items": "array",
            "shipping_address": "object",
            "notes": "string",
            "coupon_code": "string",
        },
        valid_example={
            "customer_id": "cust_abc123",
            "items": [{"product_id": "prod_1", "quantity": 2}],
            "shipping_address": {"line1": "123 Main St", "city": "Portland", "zip": "97201"},
        },
    ),
    _spec(
        api_name="Add Cart Item",
        http_method="POST",
        endpoint="/api/cart/items",
        required_fields=["product_id", "quantity"],
        optional_fields=["variant_id", "notes"],
        field_types={
            "product_id": "string",
            "quantity": "integer",
            "variant_id": "string",
            "notes": "string",
        },
        valid_example={
            "product_id": "prod_abc123",
            "quantity": 1,
        },
    ),
    _spec(
        api_name="Process Payment",
        http_method="POST",
        endpoint="/api/payments",
        required_fields=["order_id", "amount", "currency", "payment_method"],
        optional_fields=["billing_email"],
        field_types={
            "order_id": "string",
            "amount": "float",
            "currency": "enum:usd,eur,gbp,inr",
            "payment_method": "enum:card,bank_transfer,wallet",
            "billing_email": "email",
        },
        valid_example={
            "order_id": "ord_abc123",
            "amount": 49.99,
            "currency": "usd",
            "payment_method": "card",
        },
    ),
    _spec(
        api_name="Apply Coupon",
        http_method="POST",
        endpoint="/api/cart/coupon",
        required_fields=["coupon_code", "cart_id"],
        field_types={
            "coupon_code": "string",
            "cart_id": "string",
        },
        valid_example={
            "coupon_code": "SAVE20",
            "cart_id": "cart_abc123",
        },
    ),
    _spec(
        api_name="Create Shipping Label",
        http_method="POST",
        endpoint="/api/shipping/labels",
        required_fields=["order_id", "carrier", "weight"],
        optional_fields=["insurance", "signature_required"],
        field_types={
            "order_id": "string",
            "carrier": "enum:usps,fedex,ups,dhl",
            "weight": "float",
            "insurance": "boolean",
            "signature_required": "boolean",
        },
        valid_example={
            "order_id": "ord_abc123",
            "carrier": "usps",
            "weight": 2.5,
        },
    ),
]

# =========================================================================
# Domain 6: Calendar and Auth
# =========================================================================

CALENDAR_AUTH_SPECS = [
    _spec(
        api_name="Create Event",
        http_method="POST",
        endpoint="/api/calendar/events",
        required_fields=["title", "start_time", "end_time"],
        optional_fields=["description", "location", "attendees", "recurrence"],
        field_types={
            "title": "string",
            "start_time": "datetime",
            "end_time": "datetime",
            "description": "string",
            "location": "string",
            "attendees": "array",
            "recurrence": "enum:none,daily,weekly,monthly",
        },
        valid_example={
            "title": "Team Standup",
            "start_time": "2026-04-05T09:00:00Z",
            "end_time": "2026-04-05T09:30:00Z",
        },
    ),
    _spec(
        api_name="OAuth Token Request",
        http_method="POST",
        endpoint="/oauth/token",
        required_fields=["grant_type", "client_id", "client_secret"],
        optional_fields=["scope", "redirect_uri"],
        field_types={
            "grant_type": "enum:authorization_code,client_credentials,refresh_token",
            "client_id": "string",
            "client_secret": "string",
            "scope": "string",
            "redirect_uri": "url",
        },
        valid_example={
            "grant_type": "client_credentials",
            "client_id": "app_abc123",
            "client_secret": "secret_xyz789",
        },
        required_headers={
            "Content-Type": "application/json",
        },
    ),
    _spec(
        api_name="Create API Key",
        http_method="POST",
        endpoint="/api/keys",
        required_fields=["name", "permissions"],
        optional_fields=["expires_at"],
        field_types={
            "name": "string",
            "permissions": "array",
            "expires_at": "datetime",
        },
        valid_example={
            "name": "production-key",
            "permissions": ["read", "write"],
        },
    ),
    _spec(
        api_name="Invite User",
        http_method="POST",
        endpoint="/api/teams/{team_id}/invites",
        required_fields=["team_id", "email", "role"],
        optional_fields=["message"],
        field_types={
            "team_id": "string",
            "email": "email",
            "role": "enum:admin,member,viewer",
            "message": "string",
        },
        valid_example={
            "team_id": "team_abc123",
            "email": "newuser@example.com",
            "role": "member",
        },
    ),
    _spec(
        api_name="Update Permissions",
        http_method="PUT",
        endpoint="/api/users/{user_id}/permissions",
        required_fields=["user_id", "permissions"],
        optional_fields=["effective_from"],
        field_types={
            "user_id": "string",
            "permissions": "array",
            "effective_from": "datetime",
        },
        valid_example={
            "user_id": "usr_abc123",
            "permissions": ["read", "write", "admin"],
        },
    ),
]


# All 30 specs in a single flat list
ALL_SPECS = (
    PAYMENT_SPECS
    + USER_SPECS
    + CONTENT_SPECS
    + MESSAGING_SPECS
    + ECOMMERCE_SPECS
    + CALENDAR_AUTH_SPECS
)


def get_random_spec(rng) -> Dict[str, Any]:
    """Pick a random spec using the provided RNG instance."""
    return rng.choice(ALL_SPECS)
