from src.prompts import build_user_prompt

def test_prompt_budget_and_markers():
    chunks = [
        {"text": "A"*2000, "id":"a","start":0,"end":2000},
        {"text": "B"*2000, "id":"b","start":0,"end":2000},
    ]
    up = build_user_prompt("Q?", chunks, max_chars=1000)
    # budgeted inclusion (rough cap)
    assert up.count("A") <= 600 and up.count("B") <= 600
    # citation markers included
    assert "[1]" in up and "[2]" in up
