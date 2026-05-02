
import time
from agent.model_selector import classify_message

# Labeled examples (message, expected_task_type)
# Sourced from test suite plus additional edge cases
LABELED_EXAMPLES = [
    ("fix the crash in main.py line 42", "code"),
    ("this function `def foo()` has a bug", "code"),
    ("write a professional email to the team", "writing"),
    ("analyze the sales data trends from Q1", "analysis"),
    ("write a creative story about a robot", "creative"),
    ("explain why this architecture would fail at scale", "reasoning"),
    ("hello there", "general"),
    ("quick what time is it", "general"),  # urgency realtime
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
]

def run_benchmark():
    correct = 0
    total_latency = 0.0
    for msg, expected in LABELED_EXAMPLES:
        start = time.perf_counter()
        result = classify_message(msg)
        elapsed = time.perf_counter() - start
        total_latency += elapsed
        predicted = result.get("task_type", "unknown")
        if predicted == expected:
            correct += 1
        else:
            print(f"MISMATCH: '{msg}' -> predicted {predicted}, expected {expected}")
    accuracy = correct / len(LABELED_EXAMPLES)
    avg_latency_ms = total_latency / len(LABELED_EXAMPLES) * 1000
    return accuracy, avg_latency_ms

if __name__ == "__main__":
    acc, lat = run_benchmark()
    print(f"ACCURACY: {acc:.3f}")
    print(f"AVG_LATENCY_MS: {lat:.3f}")
