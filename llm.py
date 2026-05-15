import os
from groq import Groq
import json
import re
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found in .env")

client = Groq(api_key=api_key)


def extract_filters(query: str) -> dict:
    prompt = f"""
You are an AI that extracts shopping filters from user queries for a fashion dataset.

Available categories (use ONLY these, case insensitive match):
formal shoes, tshirts, trousers, tops, sports shoes, dresses, handbags, heels,
casual shoes, shirts, watches, flip flops, skirts, shorts, caps, sandals,
jeans, flats, capris, track pants, sweaters, sweatshirts, jackets, belts,
tracksuits, backpacks, tunics, sports sandals, bra

Category mapping rules:
- "tshirt", "t-shirt", "tee" → "tshirts"
- "sneakers", "running shoes", "gym shoes" → "sports shoes"
- "loafers", "moccasins" → "casual shoes"
- "blazer" → "jackets"
- "hoodie" → "sweatshirts"
- "pants", "chinos", "slacks" → "trousers"
- "leggings", "jeggings" → "capris"
- "slipper", "flip flop", "chappal" → "flip flops"
- "bag", "purse", "clutch", "tote" → "handbags"
- "backpack", "rucksack" → "backpacks"
- "hat", "beanie", "cap" → "caps"
- "gown", "frock", "maxi" → "dresses"
- "blouse", "kurti", "top" → "tops"
- "kurta", "tunic" → "tunics"
- "pump", "stiletto" → "heels"
- "oxford", "derby", "formal" → "formal shoes"
- "joggers", "track" → "track pants"
- "bra", "innerwear", "lingerie" → "bra"

Available colors (use ONLY these):
black, red, yellow, navy blue, brown, purple, white, blue, grey, pink,
green, silver, steel, gold, copper, magenta, maroon, beige, cream,
orange, olive, tan, grey melange, teal

Color mapping rules:
- "navy", "dark blue" → "navy blue"
- "gray" → "grey"
- "off white", "ivory" → "cream"
- "wine", "burgundy" → "maroon"
- "khaki" → "tan"

Extract:
- color: map to closest available color or null
- max_price: number only or null
- category: map to closest available category or null
- brand: extract as-is in lowercase or null

Return ONLY valid JSON. No explanation, no markdown, no backticks.

Examples:

Query: "black tshirt under 500"
Output:
{{"color": "black", "max_price": 500, "category": "tshirts", "brand": null}}

Query: "nike red sneakers under 3000"
Output:
{{"color": "red", "max_price": 3000, "category": "sports shoes", "brand": "nike"}}

Query: "blue skirts"
Output:
{{"color": "blue", "max_price": null, "category": "skirts", "brand": null}}

Query: "puma sweatshirt grey under 2000"
Output:
{{"color": "grey", "max_price": 2000, "category": "sweatshirts", "brand": "puma"}}

Query: "{query}"
Output:
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
        )

        text = response.choices[0].message.content.strip()
        print("LLM RAW:", text)

        text = re.sub(r"```(?:json)?", "", text).strip("`").strip()

        try:
            result = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                result = json.loads(match.group())
            else:
                raise ValueError("No JSON found in LLM response")

        cleaned = {
            "color":     result.get("color") or None,
            "max_price": float(result["max_price"]) if result.get("max_price") else None,
            "category":  result.get("category") or None,
            "brand":     result.get("brand") or None,
        }

        print("Extracted filters:", cleaned)
        return cleaned

    except Exception as e:
        print("LLM error:", e)

    return {"color": None, "max_price": None, "category": None, "brand": None}