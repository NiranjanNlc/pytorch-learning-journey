# When Not to Use Machine Learning

## When Simple Rule-Based Systems Suffice
- If a task can be solved with a simple rule-based system, machine learning (ML) should be avoided to prevent overcomplication.
- **Example**: Tasks like a calculator, where inputs map to outputs via clear, deterministic rules (e.g., `2 + 3 = 5`), do not require ML.

## Insufficient or Non-Diverse Data
- ML requires sufficient and diverse data to make reliable inferences. Without enough variation in the data, ML models cannot generalize effectively.
- **Example**: In a driving system, if data lacks diversity (e.g., only daytime conditions), the model will fail in scenarios like nighttime or rain.

## Error-Intolerant Scenarios
- ML relies on probabilistic predictions, which may lead to errors. It is unsuitable for applications where errors are unacceptable or where absolute certainty is required.
- **Example**: In critical systems like medical equipment or aerospace controls, ML’s probabilistic nature makes it risky, as it cannot guarantee completely correct outcomes.
