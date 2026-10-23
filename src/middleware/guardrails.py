

def input_guardrail(inputs: dict) -> dict:
    """
    Checks that the question is on Cambodia
    """
    question = inputs.get("question","").lower()
    blocked_topics = ["personal", "personal data","private","ignore previous","system:"]
    for topic in blocked_topics:
        if topic in question:
            raise ValueError("The agent was blocked for security.")
        
    return input_guardrail
    
   