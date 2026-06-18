"""
Generate realistic demo CSVs so the system runs before the official Ministry
files are dropped in. When you receive the real
`consumer_complaints(in).csv` and `establishments(in).csv`, just place them in
the ../data/ folder and the loader's flexible column mapping will pick them up.

Run:  python generate_sample_data.py
"""
import csv
import random
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
DATA.mkdir(exist_ok=True)

PROVINCES = ["Beirut", "Mount Lebanon", "North", "South", "Bekaa", "Nabatieh"]

ESTABLISHMENTS = [
    # name, zone, province, violations, open_complaints
    ("Al Madina Supermarket", "RED", "Beirut", 6, 4),
    ("Beirut Fresh Butchery", "RED", "Beirut", 5, 3),
    ("Cedar Grill Restaurant", "YELLOW", "Mount Lebanon", 2, 1),
    ("Sea Breeze Cafe", "GREEN", "South", 0, 0),
    ("Tripoli Mega Market", "YELLOW", "North", 3, 2),
    ("Zahle Family Bakery", "GREEN", "Bekaa", 1, 0),
    ("Nabatieh Mini Market", "RED", "Nabatieh", 4, 3),
    ("Hamra Electronics Shop", "GREEN", "Beirut", 0, 1),
    ("Jounieh Seafood House", "YELLOW", "Mount Lebanon", 2, 2),
    ("Saida Pharmacy Plus", "GREEN", "South", 1, 0),
]

# (subject, message, purchase_place, citizen_priority)
COMPLAINTS = [
    ("Food poisoning after meal", "My whole family got food poisoning and my child was hospitalized after eating expired meat here.", "Beirut Fresh Butchery", "high"),
    ("Expired products on shelf", "I bought milk that was two weeks past its expiry date, it was rotten and spoiled.", "Al Madina Supermarket", "medium"),
    ("Overcharged for subsidized goods", "They charged me double the official price for subsidized bread.", "Tripoli Mega Market", "high"),
    ("Rude staff and no refund", "The employee was extremely rude and refused to give me a refund for a broken item.", "Sea Breeze Cafe", "low"),
    ("Cockroaches in the kitchen", "I saw cockroaches and rats near the food preparation area, very unhygienic.", "Nabatieh Mini Market", "medium"),
    ("Fake branded product", "They sold me a counterfeit phone charger claiming it was original, total scam.", "Hamra Electronics Shop", "medium"),
    ("Operating without a license", "This place is operating without any health permit or license.", "Cedar Grill Restaurant", "low"),
    ("Defective blender", "The blender I bought stopped working after one day, poor quality.", "Hamra Electronics Shop", "low"),
    ("Spoiled seafood made me sick", "The fish was spoiled and I got severe diarrhea and had to go to the hospital.", "Jounieh Seafood House", "medium"),
    ("Wrong price at checkout", "The price at the register was higher than the price tag on the shelf.", "Zahle Family Bakery", "low"),
    ("Repeated hygiene problems", "Again the bathroom and kitchen are filthy, this keeps happening every time.", "Al Madina Supermarket", "low"),
    ("Expired medicine sold", "They sold me expired medicine, this is dangerous and unsafe.", "Saida Pharmacy Plus", "high"),
    ("Slow service complaint", "I waited over an hour and the staff ignored me completely.", "Cedar Grill Restaurant", "low"),
    ("Moldy bread sold as fresh", "The bread was moldy and contaminated but sold as fresh baked.", "Zahle Family Bakery", "medium"),
    ("Severe fraud with payment", "They took my money but never delivered the product, clear fraud.", "Tripoli Mega Market", "high"),
    ("Bad smell from meat section", "There is a terrible smell and the meat looks spoiled and unsafe.", "Nabatieh Mini Market", "medium"),
    ("Damaged packaging", "The packaging was torn and the product inside was damaged.", "Sea Breeze Cafe", "low"),
    ("Price gouging on basics", "Exorbitant prices, they are gouging customers on basic goods.", "Al Madina Supermarket", "medium"),
    ("Unhygienic food handling", "Staff handled raw meat without gloves right next to ready food.", "Beirut Fresh Butchery", "high"),
    ("Minor billing question", "I think I was charged a small extra fee, can you check.", "Saida Pharmacy Plus", "low"),
]


def write_establishments():
    with open(DATA / "establishments(in).csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["establishment_id", "name", "zone", "province",
                    "violations", "open_complaints"])
        for i, (name, zone, prov, v, oc) in enumerate(ESTABLISHMENTS, 1):
            w.writerow([f"E{i:03d}", name, zone, prov, v, oc])


def write_complaints():
    with open(DATA / "consumer_complaints(in).csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["complaint_id", "subject", "message", "province",
                    "purchase_place", "citizen_priority", "status"])
        est_prov = {name: prov for name, _, prov, _, _ in ESTABLISHMENTS}
        for i, (subj, msg, place, prio) in enumerate(COMPLAINTS, 1):
            prov = est_prov.get(place, random.choice(PROVINCES))
            w.writerow([f"C{i:04d}", subj, msg, prov, place, prio, "New"])


if __name__ == "__main__":
    write_establishments()
    write_complaints()
    print(f"Wrote sample data to {DATA}")
