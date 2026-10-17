"""Fill cause_group/notes in the error sample (author-drafted, to be reviewed)."""
import pandas as pd
from src.config import RESULTS_DIR

TO, WP, PE, IC, LQ, MM = ("topic_overlap", "world_politics_boundary",
    "political_content_labelled_entertainment", "insufficient_context",
    "label_questionable", "model_miss")

ANN = {
 "1c6c45dcbe": (LQ, "Vegas food/travel piece labelled HEALTH; reads as lifestyle"),
 "263d904e06": (TO, "happiness essay; wellness or lifestyle both fit"),
 "e75fa45611": (LQ, "fashion tips labelled HEALTH; reads as lifestyle"),
 "786fc3095f": (WP, "US factory closures labelled WORLD; domestic story, POLITICS seems reasonable"),
 "92937fa944": (WP, "Indonesian court case labelled POLITICS; WORLD seems at least as fitting"),
 "19ed52f922": (IC, "celebrity tweet, empty description, nothing to go on"),
 "62c74eeff7": (TO, "skin tips; health and lifestyle both fit"),
 "c3e64ca15b": (WP, "India train crash labelled POLITICS; reads as world news"),
 "47bde7d7a5": (TO, "charter schools and student data; education and business both fit"),
 "2860cf8fe7": (TO, "selfies and self-actualization; wellness or lifestyle"),
 "0e71b203b5": (TO, "athlete on cancel culture/COVID; sports and politics both fit"),
 "682cbc91d8": (WP, "poll on Merkel vs Trump on foreign affairs; world or politics"),
 "45b22f4013": (TO, "celebrity rehab; entertainment and health both fit, description minimal"),
 "cd18fa212a": (MM, "iPhone apps story; TECH label fits, prediction of HEALTH does not"),
 "7be1a0d0ac": (WP, "NZ PM on gun laws after attack; foreign leader, world or politics"),
 "f228bde7bd": (PE, "TV host's commentary on the Trump administration"),
 "79b234d284": (WP, "US senator on Hillary Clinton's memoir; WORLD label looks odd, POLITICS fits"),
 "5b499623ba": (TO, "Super Bowl ad; entertainment and sports both fit"),
 "6923dd576f": (PE, "actor on Trump; political content about an entertainer"),
 "d2e7d135fa": (PE, "TV pundit on immigration; political content, media figure"),
 "1dd3dcbf34": (PE, "late-night host on a GOP policy"),
 "f26694e6f4": (WP, "White House aide on Iran; US politics and foreign policy both"),
 "b0b2c78c9c": (PE, "TV guest remarks on immigration/white supremacy"),
 "51cf788a0c": (TO, "wildfire's local economic impact; business and politics label"),
 "2d4399879b": (MM, "fans' tribute to a singer; ENTERTAINMENT fits, LIFESTYLE does not"),
 "a9599ae723": (TO, "sports joke about the election; sports/politics/entertainment"),
 "07613dcf07": (TO, "neo-Nazi merchandise at a record shop; politics vs entertainment"),
 "4d2d966e56": (IC, "justice dressing up for a player; description is just 'All Rise!'"),
 "deb9636885": (TO, "SWIFT cuts off North Korean banks; world and business both fit"),
 "8f6a1e2741": (LQ, "president's daughter graduating; POLITICS label looks odd"),
}

path = RESULTS_DIR / "error_analysis_sample.csv"
d = pd.read_csv(path, dtype={"id": str}).fillna("")
assert set(d["id"]) == set(ANN), "ids in file and annotations differ"
d["cause_group"] = d["id"].map(lambda i: ANN[i][0])
d["notes"] = d["id"].map(lambda i: ANN[i][1])
d.to_csv(path, index=False, encoding="utf-8")
print(d.groupby(["cause_group", "split"]).size().unstack(fill_value=0))
print(len(d), "rows annotated")