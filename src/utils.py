import json
import base64


per_million_usd = {
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    "gpt-4o-mini": {"input": 0.075, "output": 0.30},
    "gpt-4o": {"input": 0.075, "output": 0.30},
}


def calculate_cost(
    prompt_tokens,
    candidates_token_count,
    context_tokens=0,
    model_name="gemini-1.5-flash",
):
    pricing = per_million_usd.get(model_name, None)
    if pricing is None:
        print(f"Unsupported model: {model_name}")
        return None  #

    # Pricing per 1 million tokens
    input_price_per_million = pricing["input"]  # USD
    output_price_per_million = pricing["output"]  # USD

    # Convert token counts to millions
    prompt_tokens_million = prompt_tokens / 1_000_000
    candidates_token_count_million = candidates_token_count / 1_000_000
    input_cost = prompt_tokens_million * input_price_per_million
    output_cost = candidates_token_count_million * output_price_per_million
    total_cost = input_cost + output_cost
    return total_cost


def encode_image(image_path):
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def dump_data_to_json(data, filename):
    """Dumps data to a JSON file."""
    try:
        with open(filename, "w") as file:
            json.dump(data, file, indent=4)
        return "Data has been dumped to the file successfully."
    except IOError as e:
        return f"An error occurred while writing to the file: {str(e)}"


# dump_data_to_json(response_req["response"], file.replace(".jpg", ".json"))
