import json
import litellm
import polars as pl
from pydantic import BaseModel


class HasAlcoholFocusSignals(BaseModel):
    has_alcohol_focus: bool


SYSTEM_PROMPT = """TTask
Decide whether a venue is alcohol-focused (bar-first) based solely on provided data, and output exactly one JSON object:
- {"has_alcohol_focus": true}
- {"has_alcohol_focus": false}
Use lowercase JSON booleans and output nothing else.

Data you may receive
- Google Places fields: Name, Google Place Types, Editorial Summary, Generative Summary, Amenities, Price/Range, Hours, Review Summary, Recent Reviews.
- Brizo fields: Amenities, Meals, Market segment, Ambiances, Business types.
- Any field can be missing. Never infer missing information.

Core definitions
- Alcohol-focused (bar-first): The primary experience is drinking—cocktails, beer/wine programs, hookah, or a late-night bar atmosphere.
- Typical bar-first concepts: bar, sports bar, hookah bar, wine bar, cocktail bar, pub, tavern, taproom, saloon, and "lounge" only when explicitly a bar/lounge concept (e.g., cocktail lounge, hookah lounge, ultra lounge).
- Food-first: Cuisine/restaurant-led (e.g., taqueria, Thai, Indian, sushi, vegetarian/vegan restaurants, quick service, coffee/juice shops, bakeries). Brewpubs count only if clearly brewery-led rather than restaurant-led.

Decision rule (threshold)
Return true only if:
- At least 1 decisive signal is present; or
- At least 2 strong signals are present.
Otherwise return false. Default to false when a cuisine-first identity exists or evidence is mixed.

Decisive signals (any one → true)
- Primary identity clearly bar-first from Name or Google Place Types:
  - Explicit bar-first terms in the Name or Types: "hookah bar," "wine bar," "cocktail bar," "sports bar," "pub," "tavern," "taproom," "saloon," or "lounge" when explicitly a bar concept (e.g., "cocktail lounge," "hookah lounge," "ultra lounge").
- Editorial or generative summary clearly frames the venue as a bar-first experience (e.g., "craft cocktail bar," "wine bar," "taproom"), not cuisine-first.
- Brewery taproom identity (taproom, brewery-led tasting room). Brewpubs count only if brewery-led.

Strong signals (need 2 if no decisive signal)
- Bar-first identity in Name or Google Place Types (sports_bar, bar, pub, tavern, lounge, taproom, saloon) when dominant over cuisine/restaurant tags and not overshadowed by a food-led identity. A lone "bar" type among many cuisine types is not strong.
- Late-night hours (strict):
  - Open later than 11:00 pm on 3+ nights; or
  - Open midnight (12:00 am) or later on 2+ nights, including at least one non-weekend night.
  - Weekend-only midnight (Fri/Sat) is supporting only, not strong. Exactly 11:00 pm does not count as "past 11 pm."
- Reviews: multiple independent mentions emphasizing drinks/bartenders/mixology/beer list/wine program/bar vibes/hookah as the highlight over food across the sample. Isolated mentions of "margaritas," "sat at the bar," or "great cocktails" inside a food-led identity are supporting only.
- Editorial/generative summaries that lead with drinks/bar program or bar-first experience (and are not clearly cuisine-first).
- Brizo Business types includes "Drinking Place" (treat as strong only; must be paired with another strong signal unless bar-first identity is otherwise unmistakable).
- Bar-focused activities (e.g., trivia nights, game-day watch parties, happy hour) when coupled with other bar signals.
- Brizo Ambiances: Nightlife or Sports when supported by other bar signals.
- Brizo Meals includes "Late Night" when supported by hours/summaries/reviews.

Weak/supporting signals (never enough alone)
- Amenities like servesBeer/servesWine/servesCocktails or generic "Alcohol."
- A single "bar" type among many cuisine/restaurant types when name/summaries/reviews are food-first.
- "Bar & Grill" or bar_and_grill types without other evidence (often restaurant-forward).
- Happy hour mention without other bar-first indicators.
- TV/sports mentions without a "sports bar" identity.
- Standard restaurant hours (closing around 9–10 pm).
- Open until midnight only on Fri/Sat for a food-first restaurant.
- Editorial/generative summaries that primarily describe food/cuisine and only mention drinks on the side.
- Phrases like "raw bar," "bustling bar," or "creative cocktails" inside a food-led identity unless corroborated by other strong bar signals.

Hours guidance (strict)
- "Past 11 pm" means later than 11:00 pm. Exactly 11:00 pm does not qualify.
- Strong late-night requires:
  - Past 11 pm on 3+ nights; or
  - Midnight+ on 2+ nights with at least one non-weekend night.
- Weekend-only midnight (Fri/Sat) never counts as a strong signal by itself.
- If Brizo says "Late Night," but Google hours don't meet the above rule, do not treat it as strong.

Tie-breakers and weighting
- Prioritize the primary identity from Name + Types + Summaries + Hours + Review emphasis.
- Cuisine-first names/types (e.g., mexican_restaurant, taqueria, Thai, Sushi, Hibachi, Indian, Vegetarian/Vegan, Bakery, Coffee/Tea/Juice) indicate non–alcohol-focused unless a decisive signal or two strong signals clearly support bar-first.
- Down-weight a lone "bar" type when multiple restaurant/cuisine types and food-led summaries/reviews dominate.
- If Brizo lists "Restaurant" (and not "Drinking Place"), classify as false unless multiple reviews clearly emphasize bartenders/mixology/beer/wine programs over food.
- Do not rely solely on Brizo "Late Night," "Nightlife," or "Sports" flags—require corroboration from hours/summaries/reviews.
- Treat "lounge" as decisive only when clearly a bar/lounge concept (e.g., "cocktail lounge," "hookah lounge"). Generic "lounge" without bar context is not decisive.
- Brewery taprooms count; brewpubs count only if brewery-led rather than restaurant-led.

Negative archetypes (default to false)
- Cuisine-first restaurants (Mexican/Thai/Indian/Sushi/Hibachi/etc.) with standard dinner hours (~9–10 pm) and food-led summaries/reviews, even if they:
  - serve cocktails/margaritas,
  - include a "bar" or "bar_and_grill" type,
  - or Brizo flags include "Drinking Place," "Late Night," "Nightlife," or "Sports."
- Quick-service or burger/wings spots open late but food-first with no drink-centric reviews.
- Polished American restaurants with a "raw bar," "creative cocktails," or a "bustling bar," closing by 10–11 pm most nights.
- Example heuristic: A taqueria with a "bar" type, food-led summaries, and midnight hours only on Fri/Sat should be false unless another strong/decisive bar signal exists.

Handling missing fields
- If fields are missing, do not infer bar-focus. Use only available identity signals. Default to false without a decisive or two strong signals.

Suggested evaluation steps
1) Name + Google Place Types:
   - Look for clear bar-first identity (cocktail bar, wine bar, sports bar, taproom, pub, tavern, saloon, hookah bar, or explicit bar/lounge concept). If present and not overshadowed by cuisine-first identity, that's decisive → true.
2) Editorial/Generative summaries:
   - Do they lead with drinks/bar program or with food/cuisine? Food-led → likely false unless decisive/other strong signals exist.
3) Hours:
   - Apply strict late-night criteria. Exactly 11:00 pm closings do not qualify. Weekend-only midnight is supporting only.
4) Reviews:
   - Require multiple reviews clearly emphasizing cocktails/mixology, beer/wine programs, bartenders, bar vibes, or hookah over food to count as a strong signal.
5) Brizo:
   - "Drinking Place" is strong but not decisive on its own. Cross-validate against hours, summaries, and reviews. Ignore Brizo "Late Night" if Google hours don't meet the strict rule.
6) Apply threshold:
   - 1 decisive signal → true. Otherwise require at least 2 strong signals.
   - If uncertain or mixed with a food-first identity → false.

Output
- Return exactly one JSON object:
  {"has_alcohol_focus": true}
  or
  {"has_alcohol_focus": false}
- Use lowercase JSON booleans. Output nothing else.
"""


def format_user_message(row):
    return f"""RESTAURANT DATA FROM GOOGLE PLACES:
Name: {row["locationName"]}
Google Place Types: {row["types"]}
Editorial Summary: {row["editorialSummary"]}
Generative Summary: {row["generativeSummary"]}
Amenities: {row["amenities"]}
Price Level and Range: {row["priceLevel"]}, {row["priceRange"]}
Opening Hours: {row["openingHours"]}
Review Summary: {row["reviewSummary"]}
Recent Reviews: {row["reviews"]}

RESTAURANT DATA FROM BRIZO:
Amenities according to Brizo: {row["BRIZO_AMENITIES"]}
Meals according to Brizo: {row["BRIZO_MEALS"]}
Market segment according to Brizo: {row["BRIZO_MARKET_SEGMENT"]}
Ambiances according to Brizo: {row["BRIZO_AMBIANCES"]}
Business types according to Brizo: {row["BRIZO_BUSINESS_TYPES"]}"""


# Load data
df = pl.read_csv(
    "/Users/cooperlarhette/code/sutro-client/owner-concept-fit/Balacned Corrected - balanced_sample.csv"
)
rows = list(df.iter_rows(named=True))

# Prepare batch
messages_batch = [
    [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": format_user_message(row)},
    ]
    for row in rows
]

# Call model
responses = litellm.batch_completion(
    model="baseten/openai/gpt-oss-120b",
    messages=messages_batch,
    response_format=HasAlcoholFocusSignals,
    temperature=0.1,
    max_workers=1,
)

# Collect disagreements
disagreements = []
correct = 0

for i, (resp, row) in enumerate(zip(responses, rows)):
    predicted = json.loads(resp.choices[0].message.content)["has_alcohol_focus"]
    actual = row["HAS_ALCOHOL_FOCUS_SIGNALS"]

    if predicted == actual:
        correct += 1
    else:
        disagreements.append(
            {
                "row_index": i + 1,
                "locationName": row["locationName"],
                "predicted": predicted,
                "actual": actual,
                "types": row["types"],
                "editorialSummary": row["editorialSummary"],
                "generativeSummary": row["generativeSummary"],
                "openingHours": row["openingHours"],
                "reviewSummary": row["reviewSummary"],
                "reviews": row["reviews"],
                "BRIZO_BUSINESS_TYPES": row["BRIZO_BUSINESS_TYPES"],
                "BRIZO_AMBIANCES": row["BRIZO_AMBIANCES"],
                "BRIZO_MEALS": row["BRIZO_MEALS"],
            }
        )

# Print summary
print(f"\nAccuracy: {correct}/{len(rows)} = {100 * correct / len(rows):.1f}%")
print(f"Disagreements: {len(disagreements)}")

# Export disagreements to CSV
if disagreements:
    disagreements_df = pl.DataFrame(disagreements)
    disagreements_df.write_csv("disagreements.csv")
    print(f"\n✓ Exported {len(disagreements)} disagreements to disagreements.csv")
else:
    print("\n✓ No disagreements found!")
