"""Multi-step reasoning eval tasks.

These are complex tasks that require sequential thinking, proof construction,
and multi-step logical deduction. Simple prompts (hello world, 2+2) pass on
any backend -- these tasks specifically test content/thinking separation
under heavy reasoning load.
"""


def get_tasks() -> list[dict]:
    return [
        {
            "name": "ball_weighing_proof",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "You have 12 balls, one of which is either heavier or lighter than the rest. "
                        "Using a balance scale exactly 3 times, determine which ball is different AND "
                        "whether it is heavier or lighter. Provide a complete decision tree covering "
                        "all possible outcomes. Show your reasoning step by step."
                    ),
                }
            ],
        },
        {
            "name": "river_crossing_variant",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "A farmer needs to cross a river with a wolf, a goat, a cabbage, and a chicken. "
                        "The boat can carry the farmer and at most two items. The wolf will eat the goat "
                        "if left alone, the goat will eat the cabbage if left alone, and the chicken will "
                        "eat the cabbage if left alone. The wolf and chicken can coexist. Find the minimum "
                        "number of crossings and prove it is optimal. List every state transition."
                    ),
                }
            ],
        },
        {
            "name": "probability_chain",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "A hospital test for a rare disease (prevalence: 1 in 10,000) has 99% sensitivity "
                        "and 95% specificity. A patient tests positive. They take a second independent test "
                        "(98% sensitivity, 99% specificity) and also test positive. "
                        "Calculate the probability they actually have the disease after both tests. "
                        "Show all Bayesian updates step by step, including the intermediate posterior "
                        "that becomes the prior for the second test."
                    ),
                }
            ],
        },
        {
            "name": "constraint_satisfaction",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Schedule 6 university courses (A-F) across 3 time slots (Morning, Afternoon, Evening) "
                        "and 2 rooms (R1, R2) subject to these constraints:\n"
                        "1. A and B cannot be in the same time slot\n"
                        "2. C must be in the Morning\n"
                        "3. D and E must be in the same time slot but different rooms\n"
                        "4. F cannot be in Room R1\n"
                        "5. A must be before D (earlier time slot)\n"
                        "6. B and F cannot be in adjacent time slots\n"
                        "Find ALL valid schedules. Prove no others exist by systematic elimination."
                    ),
                }
            ],
        },
    ]
