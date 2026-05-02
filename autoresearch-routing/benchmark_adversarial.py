import time
from agent.model_selector import classify_message

# Adversarial edge cases — designed to trick keyword classifiers
LABELED_EXAMPLES = [
    # Previous 46 (full suite from expanded benchmark)
    ("fix the crash in main.py line 42", "code"),
    ("this function `def foo()` has a bug", "code"),
    ("write a professional email to the team", "writing"),
    ("analyze the sales data trends from Q1", "analysis"),
    ("write a creative story about a robot", "creative"),
    ("explain why this architecture would fail at scale", "reasoning"),
    ("hello there", "general"),
    ("quick what time is it", "general"),
    ("perform a comprehensive system-wide architecture audit of the entire codebase", "reasoning"),
    ("debug the error", "code"),
    ("what causes the error", "reasoning"),
    ("create a Python script to parse logs", "code"),
    ("draft a proposal for the new feature", "writing"),
    ("compare the performance of algorithm A vs B", "reasoning"),
    ("refactor the legacy module", "code"),
    ("review the metrics dashboard", "analysis"),
    ("design a new logo", "creative"),
    ("how does this algorithm work?", "reasoning"),
    ("implement a cache layer", "code"),
    ("summarize the quarterly report", "writing"),
    ("why does the test fail on CI but not locally?", "reasoning"),
    ("the build is broken, fix it", "code"),
    ("what is the root cause of the memory leak?", "reasoning"),
    ("optimize the database query performance", "reasoning"),
    ("write a migration script for the schema change", "code"),
    ("why is the API returning 500 errors?", "reasoning"),
    ("debug why the container keeps restarting", "code"),
    ("explain the difference between REST and GraphQL", "reasoning"),
    ("review the PR for security vulnerabilities", "code"),
    ("can you check the test coverage report?", "analysis"),
    ("write documentation for the new API endpoint", "writing"),
    ("write a test for the auth module", "code"),
    ("create a README for this project", "writing"),
    ("write a funny commit message", "creative"),
    ("compose a professional announcement", "writing"),
    ("generate some ideas for the product name", "creative"),
    ("what are the trends in the user engagement data?", "analysis"),
    ("evaluate whether we should migrate to Kubernetes", "reasoning"),
    ("show me the latency distribution for the last hour", "analysis"),
    ("thanks!", "general"),
    ("good morning", "general"),
    ("implement distributed tracing across all microservices with OpenTelemetry", "code"),
    ("analyze the correlation between deployment frequency and incident rate", "analysis"),
    ("redesign the authentication flow to support OAuth2 and SAML", "reasoning"),
    ("write a blog post about our migration to microservices", "writing"),
    ("why does the application crash under high concurrency?", "reasoning"),
    # ADVERSARIAL CASES
    # Code-heavy message but asking for reasoning
    ("why is the Python function slow when processing large JSON files?", "reasoning"),
    ("explain the traceback in the Django middleware", "reasoning"),
    # Writing that mentions code terms
    ("write a blog post about best practices for error handling", "writing"),
    ("compose an email explaining the deployment schedule", "writing"),
    # Analysis disguised as code
    ("what's the error rate across all endpoints?", "analysis"),
    ("show me the query execution time distribution", "analysis"),
    # Creative with code terms
    ("write a poem about debugging", "creative"),
    # Very short messages
    ("hi", "general"),
    ("help", "general"),
    # Ambiguous — intent matters more than keywords
    ("the server is down", "code"),
    ("is the server down?", "reasoning"),
    # Multi-signal
    ("research the best approach for implementing real-time monitoring in our Kubernetes cluster", "reasoning"),
    ("build a dashboard to visualize the security vulnerabilities found in the audit", "code"),
    ("summarize the findings from the performance benchmark we ran yesterday", "writing"),
    # Writing edge cases
    ("rewrite the error message to be more user-friendly", "writing"),
    ("edit the documentation", "writing"),
    # Analysis edge cases
    ("what percentage of tests pass?", "analysis"),
    ("how many errors occurred this week?", "analysis"),
]

def run_benchmark():
    correct = 0
    total_latency = 0.0
    mismatches = []
    for msg, expected in LABELED_EXAMPLES:
        start = time.perf_counter()
        result = classify_message(msg)
        elapsed = time.perf_counter() - start
        total_latency += elapsed
        predicted = result.get("task_type", "unknown")
        if predicted == expected:
            correct += 1
        else:
            mismatches.append((msg, predicted, expected))
            print(f"MISMATCH: '{msg}' -> predicted {predicted}, expected {expected}")
    accuracy = correct / len(LABELED_EXAMPLES)
    avg_latency_ms = total_latency / len(LABELED_EXAMPLES) * 1000
    print(f"\nACCURACY: {accuracy:.3f} ({correct}/{len(LABELED_EXAMPLES)})")
    print(f"AVG_LATENCY_MS: {avg_latency_ms:.3f}")
    return accuracy, avg_latency_ms, mismatches

if __name__ == "__main__":
    run_benchmark()
