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


# =========================================================================
# Domain 7: Analytics and Monitoring
# =========================================================================

ANALYTICS_SPECS = [
    _spec(
        api_name="Create Dashboard",
        http_method="POST",
        endpoint="/api/dashboards",
        required_fields=["name", "workspace_id"],
        optional_fields=["description", "layout", "shared"],
        field_types={
            "name": "string",
            "workspace_id": "string",
            "description": "string",
            "layout": "enum:grid,freeform,list",
            "shared": "boolean",
        },
        valid_example={
            "name": "API Latency Overview",
            "workspace_id": "ws_prod_001",
        },
    ),
    _spec(
        api_name="Add Metric",
        http_method="POST",
        endpoint="/api/metrics",
        required_fields=["name", "type", "value"],
        optional_fields=["tags", "timestamp", "unit"],
        field_types={
            "name": "string",
            "type": "enum:counter,gauge,histogram,summary",
            "value": "float",
            "tags": "array",
            "timestamp": "datetime",
            "unit": "string",
        },
        valid_example={
            "name": "api.request.duration",
            "type": "histogram",
            "value": 245.7,
        },
    ),
    _spec(
        api_name="Create Alert Rule",
        http_method="POST",
        endpoint="/api/alerts/rules",
        required_fields=["name", "metric", "threshold", "condition"],
        optional_fields=["description", "severity", "notification_channels"],
        field_types={
            "name": "string",
            "metric": "string",
            "threshold": "float",
            "condition": "enum:above,below,equals",
            "description": "string",
            "severity": "enum:critical,warning,info",
            "notification_channels": "array",
        },
        valid_example={
            "name": "High Latency Alert",
            "metric": "api.request.duration",
            "threshold": 500.0,
            "condition": "above",
        },
    ),
    _spec(
        api_name="Log Event",
        http_method="POST",
        endpoint="/api/logs",
        required_fields=["level", "message", "service"],
        optional_fields=["timestamp", "trace_id", "metadata"],
        field_types={
            "level": "enum:debug,info,warn,error,fatal",
            "message": "string",
            "service": "string",
            "timestamp": "datetime",
            "trace_id": "string",
            "metadata": "object",
        },
        valid_example={
            "level": "error",
            "message": "Connection timeout to database",
            "service": "payment-service",
        },
    ),
    _spec(
        api_name="Query Logs",
        http_method="POST",
        endpoint="/api/logs/search",
        required_fields=["query", "start_time", "end_time"],
        optional_fields=["limit", "service_filter", "level_filter"],
        field_types={
            "query": "string",
            "start_time": "datetime",
            "end_time": "datetime",
            "limit": "integer",
            "service_filter": "string",
            "level_filter": "enum:debug,info,warn,error,fatal",
        },
        valid_example={
            "query": "timeout OR connection refused",
            "start_time": "2026-04-01T00:00:00Z",
            "end_time": "2026-04-01T23:59:59Z",
        },
    ),
]

# =========================================================================
# Domain 8: DevOps and Infrastructure
# =========================================================================

DEVOPS_SPECS = [
    _spec(
        api_name="Create Deployment",
        http_method="POST",
        endpoint="/api/deployments",
        required_fields=["service_name", "image", "environment"],
        optional_fields=["replicas", "cpu_limit", "memory_limit", "env_vars"],
        field_types={
            "service_name": "string",
            "image": "string",
            "environment": "enum:staging,production,development",
            "replicas": "integer",
            "cpu_limit": "string",
            "memory_limit": "string",
            "env_vars": "object",
        },
        valid_example={
            "service_name": "api-gateway",
            "image": "registry.io/api-gateway:v2.1.0",
            "environment": "production",
        },
    ),
    _spec(
        api_name="Scale Service",
        http_method="PATCH",
        endpoint="/api/services/{service_id}/scale",
        required_fields=["service_id", "replicas"],
        optional_fields=["min_replicas", "max_replicas"],
        field_types={
            "service_id": "string",
            "replicas": "integer",
            "min_replicas": "integer",
            "max_replicas": "integer",
        },
        valid_example={
            "service_id": "svc_api_gateway",
            "replicas": 5,
        },
    ),
    _spec(
        api_name="Create DNS Record",
        http_method="POST",
        endpoint="/api/dns/records",
        required_fields=["domain", "type", "value"],
        optional_fields=["ttl", "priority"],
        field_types={
            "domain": "string",
            "type": "enum:A,AAAA,CNAME,MX,TXT,NS",
            "value": "string",
            "ttl": "integer",
            "priority": "integer",
        },
        valid_example={
            "domain": "api.example.com",
            "type": "A",
            "value": "203.0.113.50",
        },
    ),
    _spec(
        api_name="Add SSL Certificate",
        http_method="POST",
        endpoint="/api/certificates",
        required_fields=["domain", "certificate", "private_key"],
        optional_fields=["chain", "auto_renew"],
        field_types={
            "domain": "string",
            "certificate": "string",
            "private_key": "string",
            "chain": "string",
            "auto_renew": "boolean",
        },
        valid_example={
            "domain": "api.example.com",
            "certificate": "-----BEGIN CERTIFICATE-----\nMIIB...\n-----END CERTIFICATE-----",
            "private_key": "-----BEGIN PRIVATE KEY-----\nMIIE...\n-----END PRIVATE KEY-----",
        },
    ),
    _spec(
        api_name="Create Load Balancer",
        http_method="POST",
        endpoint="/api/load-balancers",
        required_fields=["name", "algorithm", "targets"],
        optional_fields=["health_check_path", "health_check_interval", "sticky_sessions"],
        field_types={
            "name": "string",
            "algorithm": "enum:round_robin,least_connections,ip_hash,weighted",
            "targets": "array",
            "health_check_path": "string",
            "health_check_interval": "integer",
            "sticky_sessions": "boolean",
        },
        valid_example={
            "name": "api-lb-prod",
            "algorithm": "round_robin",
            "targets": [
                {"host": "10.0.1.1", "port": 8080},
                {"host": "10.0.1.2", "port": 8080},
            ],
        },
    ),
]

# =========================================================================
# Domain 9: AI/ML APIs
# =========================================================================

AI_ML_SPECS = [
    _spec(
        api_name="Submit Inference",
        http_method="POST",
        endpoint="/api/inference",
        required_fields=["model_id", "inputs"],
        optional_fields=["parameters", "stream", "timeout"],
        field_types={
            "model_id": "string",
            "inputs": "string",
            "parameters": "object",
            "stream": "boolean",
            "timeout": "integer",
        },
        valid_example={
            "model_id": "meta-llama/Llama-3-8B-Instruct",
            "inputs": "Explain reinforcement learning in one sentence.",
        },
    ),
    _spec(
        api_name="Create Fine-tune Job",
        http_method="POST",
        endpoint="/api/fine-tune",
        required_fields=["base_model", "dataset_id", "num_epochs"],
        optional_fields=["learning_rate", "batch_size", "validation_split"],
        field_types={
            "base_model": "string",
            "dataset_id": "string",
            "num_epochs": "integer",
            "learning_rate": "float",
            "batch_size": "integer",
            "validation_split": "float",
        },
        valid_example={
            "base_model": "Qwen/Qwen2.5-0.5B",
            "dataset_id": "ds_api_debug_v1",
            "num_epochs": 3,
        },
    ),
    _spec(
        api_name="Upload Dataset",
        http_method="POST",
        endpoint="/api/datasets",
        required_fields=["name", "format", "source_url"],
        optional_fields=["description", "license", "tags"],
        field_types={
            "name": "string",
            "format": "enum:json,csv,parquet,arrow",
            "source_url": "url",
            "description": "string",
            "license": "string",
            "tags": "array",
        },
        valid_example={
            "name": "api-debug-training-v1",
            "format": "json",
            "source_url": "https://storage.example.com/datasets/api_debug.json",
        },
    ),
    _spec(
        api_name="Create Embedding",
        http_method="POST",
        endpoint="/api/embeddings",
        required_fields=["model_id", "input"],
        optional_fields=["encoding_format", "dimensions"],
        field_types={
            "model_id": "string",
            "input": "string",
            "encoding_format": "enum:float,base64",
            "dimensions": "integer",
        },
        valid_example={
            "model_id": "BAAI/bge-small-en-v1.5",
            "input": "API debugging is a critical developer skill.",
        },
    ),
    _spec(
        api_name="List Models",
        http_method="GET",
        endpoint="/api/models",
        required_fields=["task"],
        optional_fields=["library", "sort", "limit"],
        field_types={
            "task": "enum:text-generation,text-classification,embeddings,image-classification",
            "library": "string",
            "sort": "enum:downloads,likes,trending",
            "limit": "integer",
        },
        valid_example={
            "task": "text-generation",
        },
    ),
]


# All 45 specs in a single flat list
ALL_SPECS = (
    PAYMENT_SPECS
    + USER_SPECS
    + CONTENT_SPECS
    + MESSAGING_SPECS
    + ECOMMERCE_SPECS
    + CALENDAR_AUTH_SPECS
    + ANALYTICS_SPECS
    + DEVOPS_SPECS
    + AI_ML_SPECS
)


def get_random_spec(rng) -> Dict[str, Any]:
    """Pick a random spec using the provided RNG instance."""
    return rng.choice(ALL_SPECS)
