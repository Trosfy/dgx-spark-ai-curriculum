#!/usr/bin/env python3
"""
Example Tools for Domain-Specific AI Assistant

These are example tools that can be registered with the assistant.
Customize these for your specific domain!
"""

from typing import Dict, Any
import json


def calculate(expression: str) -> str:
    """
    Evaluate a mathematical expression.

    Args:
        expression: Math expression to evaluate (e.g., "2 + 2")

    Returns:
        Result as string
    """
    # Note: In production, use a safer evaluation method!
    try:
        # Only allow safe characters
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression):
            return "Error: Invalid characters in expression"

        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error: {str(e)}"


def search_docs(query: str, max_results: int = 3) -> str:
    """
    Search documentation (mock implementation).

    Args:
        query: Search query
        max_results: Maximum number of results

    Returns:
        Search results as formatted string
    """
    # Mock search results - replace with real search in production
    mock_results = {
        "s3": [
            {"title": "Creating S3 Buckets", "snippet": "Use aws s3 mb s3://bucket-name"},
            {"title": "S3 Best Practices", "snippet": "Enable versioning for data protection"},
        ],
        "lambda": [
            {"title": "Lambda Limits", "snippet": "Max timeout: 15 minutes, max memory: 10GB"},
            {"title": "Lambda Best Practices", "snippet": "Keep functions small and focused"},
        ],
        "ec2": [
            {"title": "EC2 Instance Types", "snippet": "Choose based on workload requirements"},
            {"title": "EC2 Pricing", "snippet": "On-demand, reserved, and spot instances available"},
        ],
    }

    results = []
    query_lower = query.lower()

    for key, docs in mock_results.items():
        if key in query_lower:
            results.extend(docs)

    if not results:
        results = [{"title": "General AWS", "snippet": "No specific results found"}]

    # Format results
    formatted = f"Found {min(len(results), max_results)} results:\n\n"
    for i, r in enumerate(results[:max_results], 1):
        formatted += f"{i}. **{r['title']}**\n   {r['snippet']}\n\n"

    return formatted


def get_weather(location: str) -> str:
    """
    Get weather information (mock implementation).

    Args:
        location: City or location name

    Returns:
        Weather information
    """
    # Mock weather - replace with real API in production
    return f"""Weather for {location}:
- Temperature: 72°F (22°C)
- Conditions: Partly cloudy
- Humidity: 45%

(This is mock data for demonstration)"""


def validate_json(json_string: str) -> str:
    """
    Validate a JSON string.

    Args:
        json_string: JSON to validate

    Returns:
        Validation result
    """
    try:
        parsed = json.loads(json_string)
        return f"✅ Valid JSON\nParsed structure: {type(parsed).__name__}"
    except json.JSONDecodeError as e:
        return f"❌ Invalid JSON: {str(e)}"


def estimate_cost(
    service: str,
    usage_hours: float = 720,  # Default: one month
    instance_type: str = "t3.medium"
) -> str:
    """
    Estimate AWS service cost (mock implementation).

    Args:
        service: AWS service name
        usage_hours: Hours of usage
        instance_type: Instance type (for EC2)

    Returns:
        Cost estimate
    """
    # Mock pricing
    pricing = {
        "ec2": {
            "t3.micro": 0.0104,
            "t3.medium": 0.0416,
            "t3.large": 0.0832,
        },
        "s3": 0.023,  # Per GB-month
        "lambda": 0.0000166667,  # Per GB-second
    }

    service_lower = service.lower()

    if "ec2" in service_lower:
        rate = pricing["ec2"].get(instance_type, 0.0416)
        cost = rate * usage_hours
        return f"""EC2 Cost Estimate:
- Instance: {instance_type}
- Hours: {usage_hours}
- Rate: ${rate}/hour
- **Estimated Cost: ${cost:.2f}**

Note: This is a simplified estimate. Actual costs may vary."""

    elif "s3" in service_lower:
        return f"""S3 Cost Estimate:
- Storage rate: ${pricing['s3']}/GB-month
- 100GB would cost ~$2.30/month
- Data transfer and requests add additional costs"""

    elif "lambda" in service_lower:
        return f"""Lambda Cost Estimate:
- Rate: ${pricing['lambda']}/GB-second
- Free tier: 1M requests + 400,000 GB-seconds/month
- Usually very cost-effective for sporadic workloads"""

    return f"No pricing data available for {service}"


# Tool registry helper
def get_all_tools():
    """Return all available tools with their metadata."""
    return {
        "calculate": {
            "function": calculate,
            "description": "Evaluate a mathematical expression",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression to evaluate"
                    }
                },
                "required": ["expression"]
            }
        },
        "search_docs": {
            "function": search_docs,
            "description": "Search AWS documentation",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum results to return"
                    }
                },
                "required": ["query"]
            }
        },
        "estimate_cost": {
            "function": estimate_cost,
            "description": "Estimate AWS service costs",
            "parameters": {
                "type": "object",
                "properties": {
                    "service": {
                        "type": "string",
                        "description": "AWS service name"
                    },
                    "usage_hours": {
                        "type": "number",
                        "description": "Hours of usage"
                    }
                },
                "required": ["service"]
            }
        },
        "validate_json": {
            "function": validate_json,
            "description": "Validate a JSON string",
            "parameters": {
                "type": "object",
                "properties": {
                    "json_string": {
                        "type": "string",
                        "description": "JSON to validate"
                    }
                },
                "required": ["json_string"]
            }
        },
    }


if __name__ == "__main__":
    print("Example Tools Demo")
    print("="*50)

    print("\n1. Calculate:")
    print(calculate("100 * 24 * 30"))

    print("\n2. Search Docs:")
    print(search_docs("s3 bucket"))

    print("\n3. Estimate Cost:")
    print(estimate_cost("ec2", 720, "t3.medium"))

    print("\n4. Validate JSON:")
    print(validate_json('{"valid": true}'))
