"""Live Commerce: the synthetic world the simulator streams (platforms, merchants, catalogs, customers).

Everything is generated from one seed, so a demo run, a reset and a test all see the
same merchants with the same products and prices. The world holds only what a platform
would really tell us (who sells what, where, at what price and cost); the hidden demand
truths that the engine has to learn (base rates, elasticities, shocks) live in
``simulator``.

Merchants are laid out like the real businesses: Shopify D2C brands with their own
product lines, Reliance Smart / Digital / Trends stores and JioMart sellers, and DMart
hypermarkets named by area. Customers belong to one home merchant, with their own
repurchase interval, and some of them lapse during the run (what win-back signals catch).
"""

import csv
import os
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from . import contract

# Which world the live wall runs: "shoppi" (the Shoppi merchants in examples/shoppi) or
# "india" (the original generated Shopify / Reliance / DMart network).
WORLD = os.environ.get("SB_COMMERCE_WORLD", "shoppi")
SHOPPI_DIR = Path(os.environ.get("SB_SHOPPI_DIR", Path(__file__).resolve().parents[3] / "examples" / "shoppi"))


@dataclass(frozen=True)
class Platform:
    id: str
    name: str
    kind: str
    color: str


PLATFORMS = (
    Platform("shopify", "Shopify", "D2C online stores", "#95BF47"),
    Platform("reliance", "Reliance Retail", "Omnichannel retail: Smart, Digital, Trends, JioMart", "#0A3A82"),
    Platform("dmart", "DMart", "Value hypermarkets", "#00833E"),
)
PLATFORM_SHARE = {"shopify": 0.40, "reliance": 0.35, "dmart": 0.25}
SHOPPI = Platform("shoppi", "Shoppi", "Marketplace merchants", "#E4572E")
if WORLD == "shoppi":
    PLATFORMS = (SHOPPI,)

# (city, latitude, longitude, weight for D2C and Reliance placement)
CITIES = {
    "Mumbai": (19.08, 72.88, 10), "Delhi": (28.61, 77.21, 10), "Bengaluru": (12.97, 77.59, 9),
    "Hyderabad": (17.39, 78.49, 7), "Chennai": (13.08, 80.27, 6), "Kolkata": (22.57, 88.36, 6),
    "Pune": (18.52, 73.86, 6), "Ahmedabad": (23.02, 72.57, 5), "Jaipur": (26.91, 75.79, 4),
    "Surat": (21.17, 72.83, 3), "Lucknow": (26.85, 80.95, 3), "Kanpur": (26.45, 80.33, 2),
    "Nagpur": (21.15, 79.09, 2), "Indore": (22.72, 75.86, 2), "Thane": (19.22, 72.98, 3),
    "Bhopal": (23.26, 77.41, 2), "Visakhapatnam": (17.69, 83.22, 2), "Patna": (25.59, 85.14, 2),
    "Vadodara": (22.31, 73.18, 2), "Ludhiana": (30.90, 75.86, 2), "Agra": (27.18, 78.01, 1),
    "Nashik": (20.00, 73.79, 1), "Kochi": (9.93, 76.27, 2), "Coimbatore": (11.02, 76.96, 2),
    "Chandigarh": (30.73, 76.78, 2), "Guwahati": (26.14, 91.74, 2), "Bhubaneswar": (20.30, 85.82, 1),
    "Mysuru": (12.30, 76.64, 1), "Thiruvananthapuram": (8.52, 76.94, 1), "Raipur": (21.25, 81.63, 1),
    "Panaji": (15.49, 73.83, 1), "Dehradun": (30.32, 78.03, 1), "Navi Mumbai": (19.03, 73.03, 2),
    "Gurugram": (28.46, 77.03, 3), "Noida": (28.54, 77.39, 2), "Madurai": (9.93, 78.12, 1),
    "Vijayawada": (16.51, 80.65, 1), "Ranchi": (23.34, 85.31, 1), "Amritsar": (31.63, 74.87, 1),
    "Jodhpur": (26.24, 73.02, 1), "Solapur": (17.66, 75.91, 1), "Aurangabad": (19.88, 75.34, 1),
    "Kolhapur": (16.70, 74.24, 1), "Mangaluru": (12.91, 74.86, 1), "Udaipur": (24.59, 73.71, 1),
}

# Neighbourhoods, used for store names (DMart is strongest in the west and south, like the real chain).
AREAS = {
    "Mumbai": ["Powai", "Malad", "Goregaon", "Bhandup", "Mulund", "Kandivali", "Borivali", "Andheri East",
               "Ghatkopar", "Chembur", "Vikhroli", "Dahisar", "Mira Road", "Bhayander"],
    "Thane": ["Ghodbunder Road", "Majiwada", "Wagle Estate", "Kalwa", "Dombivli", "Kalyan", "Ambernath", "Badlapur"],
    "Navi Mumbai": ["Vashi", "Nerul", "Kharghar", "Panvel", "Airoli", "Belapur", "Ulwe", "Sanpada"],
    "Pune": ["Hadapsar", "Kothrud", "Wakad", "Baner", "Kharadi", "Viman Nagar", "Pimpri", "Chinchwad",
             "Bavdhan", "Undri", "Wagholi", "Katraj", "Dhayari", "Aundh"],
    "Ahmedabad": ["Satellite", "Bopal", "Naroda", "Maninagar", "Chandkheda", "Vastral", "Gota", "Nikol",
                  "Vejalpur", "Thaltej"],
    "Surat": ["Adajan", "Vesu", "Katargam", "Varachha", "Pal", "Udhna"],
    "Vadodara": ["Alkapuri", "Gotri", "Manjalpur", "Waghodia Road", "Harni"],
    "Hyderabad": ["Kukatpally", "Ameerpet", "LB Nagar", "Dilsukhnagar", "Miyapur", "Kompally", "Gachibowli",
                  "Uppal", "Kondapur", "Nizampet", "Bachupally", "Attapur", "Mehdipatnam", "Secunderabad"],
    "Bengaluru": ["Whitefield", "Electronic City", "Bannerghatta Road", "Yelahanka", "Marathahalli",
                  "Sarjapur Road", "Hosur Road", "Kanakapura Road", "Rajajinagar", "Hebbal", "BTM Layout", "KR Puram"],
    "Chennai": ["Velachery", "Tambaram", "Porur", "Ambattur", "Perungudi", "Medavakkam"],
    "Nagpur": ["Wardha Road", "Manish Nagar", "Kamptee Road", "Hingna"],
    "Nashik": ["College Road", "Nashik Road", "Pathardi Phata"],
    "Indore": ["Vijay Nagar", "Bhawarkua", "Rau"],
    "Bhopal": ["Kolar Road", "Hoshangabad Road", "Ayodhya Bypass"],
    "Raipur": ["Telibandha", "Tatibandh"],
    "Gurugram": ["Sohna Road", "Palam Vihar", "Sector 56"],
    "Noida": ["Sector 63", "Sector 110"],
    "Jaipur": ["Mansarovar", "Jagatpura", "Vaishali Nagar"],
    "Aurangabad": ["Cidco", "Beed Bypass"],
    "Solapur": ["Hotgi Road"],
    "Kolhapur": ["Tarabai Park"],
    "Vijayawada": ["Benz Circle"],
    "Visakhapatnam": ["Gajuwaka", "MVP Colony"],
    "Mangaluru": ["Kankanady"],
}
GENERIC_AREAS = ["MG Road", "Station Road", "Civil Lines", "Main Market", "City Centre", "Ring Road", "Mall Road"]
DMART_CITIES = [c for c in AREAS]


@dataclass(frozen=True)
class Category:
    code: str
    margin: tuple[float, float]
    lead_days: tuple[int, int]
    perishable: bool
    qty_mean: float        # units per order line
    interval_days: float   # typical customer repurchase interval for stores selling it


CATEGORIES = {
    "Staples": Category("STP", (0.08, 0.14), (3, 5), False, 1.4, 7),
    "Snacks": Category("SNK", (0.15, 0.22), (3, 5), False, 1.6, 7),
    "Sweets": Category("SWT", (0.25, 0.40), (2, 4), False, 1.3, 20),
    "Dairy": Category("DRY", (0.08, 0.12), (1, 1), True, 1.5, 4),
    "Fresh": Category("FRS", (0.15, 0.25), (1, 2), True, 1.6, 4),
    "Personal Care": Category("PRC", (0.18, 0.28), (4, 7), False, 1.2, 14),
    "Household": Category("HHD", (0.15, 0.25), (4, 7), False, 1.2, 14),
    "Beauty": Category("BTY", (0.30, 0.40), (5, 10), False, 1.1, 30),
    "Fashion": Category("FSH", (0.35, 0.50), (10, 21), False, 1.1, 45),
    "Electronics": Category("ELC", (0.06, 0.12), (5, 10), False, 1.0, 120),
    "Electronics Accessories": Category("ACC", (0.25, 0.35), (7, 14), False, 1.1, 60),
    "Home": Category("HOM", (0.25, 0.40), (7, 14), False, 1.1, 45),
}
D2C_MARGIN = (0.50, 0.68)

# Retail (branded) products: (name, price in rupees[, perishable]).
RETAIL = {
    "Staples": [
        ("Aashirvaad Atta 5kg", 285), ("Fortune Sunflower Oil 1L", 155), ("India Gate Basmati Rice 5kg", 899),
        ("Daawat Rozana Basmati 5kg", 545), ("Tata Salt 1kg", 28), ("Tata Sampann Toor Dal 1kg", 185),
        ("Fortune Chakki Atta 10kg", 520), ("Saffola Gold Oil 1L", 189), ("Madhur Sugar 1kg", 52),
        ("Tata Tea Gold 500g", 310), ("Brooke Bond Red Label 500g", 290), ("Nescafe Classic 100g", 340),
        ("MDH Garam Masala 100g", 92), ("Everest Chhole Masala 100g", 78), ("Catch Turmeric Powder 200g", 66),
        ("Fortune Besan 1kg", 120), ("Rajdhani Moong Dal 1kg", 165), ("Amul Pure Ghee 1L", 650),
        ("Kohinoor Basmati Rice 1kg", 210), ("Sona Masoori Rice 10kg", 690), ("Pillsbury Maida 1kg", 58),
        ("Dhara Kachi Ghani Mustard Oil 1L", 175), ("Patanjali Cow Ghee 1L", 640), ("Tata Sampann Chana Dal 1kg", 120),
    ],
    "Snacks": [
        ("Haldiram's Aloo Bhujia 400g", 105), ("Lay's Magic Masala 52g", 20), ("Kurkure Masala Munch 90g", 20),
        ("Parle-G 800g", 85), ("Britannia Good Day 600g", 120), ("Maggi Noodles 12-pack", 168),
        ("Bikaji Bikaneri Bhujia 1kg", 260), ("Cadbury Oreo 300g", 60), ("Britannia Marie Gold 600g", 100),
        ("Sunfeast Dark Fantasy 300g", 120), ("Haldiram's Khatta Meetha 400g", 110), ("Bingo Mad Angles 90g", 20),
        ("Kissan Tomato Ketchup 950g", 135), ("Cadbury Bournvita 1kg", 445), ("Tropicana Orange 1L", 125),
        ("Thums Up 2L", 95), ("Paper Boat Aam Panna 250ml", 30), ("Act II Butter Popcorn 3-pack", 90),
        ("Too Yumm Multigrain Chips 54g", 20), ("Coca-Cola 2L", 95),
    ],
    "Sweets": [
        ("Haldiram's Soan Papdi 500g", 150), ("Haldiram's Rasgulla 1kg", 220), ("Bikanervala Kaju Katli 500g", 600),
        ("Haldiram's Gulab Jamun 1kg", 240), ("Cadbury Celebrations Gift Pack", 289), ("Ferrero Rocher 16pc", 650),
        ("Nutraj Dry Fruit Gift Box 500g", 899), ("Motichoor Ladoo 500g", 300, True), ("Fresh Kaju Katli 500g", 550, True),
        ("Mysore Pak 500g", 320, True), ("Besan Ladoo 500g", 260, True), ("Cadbury Dairy Milk Silk 150g", 175),
        ("Amul Dark Chocolate 150g", 125), ("Nutraj California Almonds 500g", 480), ("Happilo Cashews 500g", 575),
        ("Lindt Excellence 70% 100g", 330), ("Diwali Dry Fruit Hamper", 1199), ("Kaju Pista Roll 500g", 640, True),
    ],
    "Dairy": [
        ("Amul Taaza Toned Milk 1L", 68), ("Amul Gold Milk 1L", 74), ("Mother Dairy Toned Milk 1L", 62),
        ("Amul Butter 500g", 285), ("Amul Cheese Slices 200g", 150), ("Amul Masti Dahi 400g", 35),
        ("Amul Fresh Paneer 200g", 90), ("Britannia Bread 400g", 45), ("Farm Fresh Eggs 12pc", 84),
        ("Epigamia Greek Yogurt 90g", 50), ("Amul Fresh Cream 250ml", 70), ("Mother Dairy Lassi 200ml", 25),
    ],
    "Fresh": [
        ("Onion 1kg", 40), ("Potato 1kg", 32), ("Tomato 1kg", 36), ("Banana Robusta 12pc", 60),
        ("Shimla Apple 1kg", 180), ("Coriander Bunch", 15), ("Green Chilli 100g", 10), ("Ginger 250g", 35),
        ("Lady Finger 500g", 30), ("Cauliflower 1pc", 40), ("Pomegranate 1kg", 220), ("Lemon 6pc", 30),
    ],
    "Personal Care": [
        ("Dove Cream Beauty Bar 4x100g", 220), ("Lifebuoy Total Soap 4x125g", 140), ("Colgate Strong Teeth 200g", 110),
        ("Clinic Plus Shampoo 650ml", 340), ("Head & Shoulders Shampoo 340ml", 360), ("Parachute Coconut Oil 600ml", 260),
        ("Dettol Original Handwash 750ml", 160), ("Gillette Mach3 Cartridges 4pc", 450), ("Nivea Body Lotion 400ml", 399),
        ("Himalaya Neem Face Wash 150ml", 199), ("Pond's Bright Beauty Cream 50g", 210), ("Whisper Ultra XL 30pc", 380),
        ("Pampers Pants M 56pc", 899), ("Closeup Toothpaste 150g", 99), ("Vaseline Lip Care 10g", 95),
    ],
    "Household": [
        ("Surf Excel Easy Wash 1.5kg", 230), ("Ariel Matic Front Load 2kg", 549), ("Tide Plus 2kg", 270),
        ("Vim Dishwash Liquid 750ml", 175), ("Harpic Power Plus 1L", 210), ("Lizol Floor Cleaner 975ml", 229),
        ("Colin Glass Cleaner 500ml", 115), ("Good Knight Gold Flash Refill 2-pack", 175), ("Comfort Fabric Conditioner 860ml", 230),
        ("Scotch-Brite Scrub Pad 3-pack", 60), ("Hit Mosquito Spray 400ml", 225), ("Odonil Room Freshener 4-pack", 199),
        ("Exo Dishwash Bar 500g", 50), ("Garbage Bags Medium 90pc", 145),
    ],
    "Beauty": [
        ("Lakme 9to5 Primer + Matte Lipstick", 499), ("Maybelline Fit Me Foundation 30ml", 549),
        ("Mamaearth Vitamin C Face Wash 100ml", 349), ("Plum Green Tea Toner 200ml", 390),
        ("Biotique Bio Papaya Scrub 75g", 199), ("Nykaa Matte to Last Lipstick", 349), ("Lotus Safe Sun SPF 50 100g", 495),
        ("L'Oreal Paris Hyaluron Serum 30ml", 799), ("Minimalist Niacinamide 10% 30ml", 599), ("WOW Onion Hair Oil 200ml", 399),
        ("Sugar Kohl of Honour Kajal", 299), ("Forest Essentials Kumkumadi Oil 15ml", 1995), ("The Derma Co Sunscreen 50g", 499),
        ("Kama Ayurveda Rose Water 200ml", 795),
    ],
    "Fashion": [
        ("Avaasa Women's Printed Kurta", 699), ("Netplay Men's Casual Shirt", 899), ("DNMX Men's Slim Jeans", 1199),
        ("Teamspirit Track Pants", 599), ("Fig Women's Top", 499), ("Performax Dry-Fit T-shirt", 449),
        ("John Players Men's Blazer", 3999), ("Rio Girls Party Dress", 799), ("Avaasa Salwar Suit Set", 1299),
        ("Netplay Men's Chinos", 1099), ("DNMX Hoodie", 999), ("Ajile Running Shoes", 1299),
        ("Men's Cotton Vest 3-pack", 349), ("Women's Ankle Leggings", 299), ("Kids Graphic T-shirt", 199),
        ("Men's Cotton Kurta Pyjama", 999), ("Women's Art Silk Saree", 1499), ("Boys Ethnic Kurta Set", 899),
        ("Women's Palazzo Pants", 499), ("Men's Formal Trousers", 899), ("Girls Lehenga Choli", 1199),
        ("Unisex Flip Flops", 249),
    ],
    "Electronics": [
        ("Samsung Galaxy M35 5G", 16999), ("Redmi Note 14 5G", 18999), ("Apple iPhone 16 128GB", 79900),
        ("OnePlus Nord CE4", 24999), ("Samsung 43\" Crystal 4K TV", 32990), ("LG 55\" 4K Smart TV", 52990),
        ("Sony Bravia 50\" 4K TV", 64990), ("Whirlpool 265L Refrigerator", 27490), ("LG 7kg Front Load Washer", 33990),
        ("Samsung 28L Convection Microwave", 13990), ("Philips Air Fryer 4.1L", 7999), ("Prestige Induction Cooktop", 2499),
        ("Bajaj Mixer Grinder 750W", 3299), ("Voltas 1.5T Split AC", 36990), ("HP 15s Laptop i5", 54990),
        ("Lenovo IdeaPad Slim 3", 45990), ("Apple AirPods Pro 2", 24900), ("JBL Flip 6", 9999),
        ("Sony WH-1000XM5", 29990), ("Canon EOS 1500D", 36990), ("Havells Geyser 15L", 7499),
        ("Apple Watch SE", 24900), ("Samsung Galaxy Tab A9", 13999), ("Reconnect 32\" HD Smart TV", 11990),
    ],
    "Electronics Accessories": [
        ("boAt Airdopes 141", 1299), ("boAt Rockerz 255 Pro+", 1499), ("Noise ColorFit Pulse 4", 1999),
        ("Mi Power Bank 3i 20000mAh", 2199), ("Ambrane 10000mAh Power Bank", 999), ("Portronics Konnect USB-C Cable", 299),
        ("Samsung 25W Adapter", 1499), ("Apple 20W USB-C Adapter", 1900), ("SanDisk 64GB Pendrive", 549),
        ("SanDisk 128GB microSD", 1099), ("Logitech M235 Wireless Mouse", 799), ("Zebronics USB Keyboard", 499),
        ("JBL Go 3", 2999), ("realme Buds 2", 599), ("Fire-Boltt Ninja Smartwatch", 1599), ("Spigen Phone Case", 999),
    ],
    "Home": [
        ("Milton Thermosteel Bottle 1L", 899), ("Cello Opalware Dinner Set 27pc", 1499), ("Prestige Pressure Cooker 5L", 2199),
        ("Pigeon Non-stick Tawa", 649), ("Borosil Glass Set 6pc", 449), ("Philips LED Bulb 9W 4-pack", 399),
        ("Clay Diya Set 12pc", 149), ("Decorative String Lights 15m", 299), ("Bombay Dyeing Double Bedsheet", 899),
        ("Spaces Bath Towel", 499), ("Tupperware Container Set", 999), ("Steel Kadhai 2L", 799),
        ("Wipro Extension Board", 549), ("Rangoli Colours 10-pack", 199), ("Plastic Storage Box 3-pack", 349),
        ("Brass Pooja Thali", 699),
    ],
}

# D2C product lines, prefixed with the brand's name on Shopify.
D2C = {
    "Fashion": [
        ("Linen Kurta", 1499), ("Cotton Anarkali Set", 2499), ("Chikankari Kurti", 1799), ("Banarasi Silk Saree", 4999),
        ("Block Print Dupatta", 899), ("Nehru Jacket", 2199), ("Oversized Graphic Tee", 799), ("Slim Fit Chinos", 1599),
        ("Palazzo Pants", 999), ("Denim Jacket", 2499), ("Kolhapuri Chappals", 1199), ("Embroidered Juttis", 1399),
        ("Silk Kurta Pyjama Set", 2999), ("Festive Lehenga Choli", 6999), ("Cotton Shirt", 1299), ("Printed Co-ord Set", 1899),
        ("Handloom Saree", 3499), ("Jhumka Earrings", 599), ("Canvas Tote Bag", 999), ("Joggers", 1099),
        ("Ethnic Sherwani", 8999), ("Kids Kurta Set", 1199), ("Bandhani Dupatta", 1099), ("Pathani Suit", 2299),
    ],
    "Beauty": [
        ("Vitamin C Face Serum 30ml", 599), ("Kumkumadi Night Oil 15ml", 895), ("Ubtan Face Wash 100ml", 349),
        ("Onion Hair Oil 200ml", 399), ("Matte Liquid Lipstick", 469), ("Rose Water Toner 100ml", 299),
        ("Saffron Face Cream 50g", 649), ("Sunscreen SPF 50 50g", 499), ("Bhringraj Shampoo 250ml", 399),
        ("Charcoal Face Mask 100g", 449), ("Kajal Stick", 249), ("Lip Balm Duo", 299), ("Body Butter 200g", 549),
        ("Niacinamide Serum 30ml", 649), ("Hair Growth Serum 50ml", 799), ("Festive Gift Kit", 1499),
        ("Sandalwood Soap 3-pack", 399), ("Under Eye Cream 15ml", 499), ("Nail Paint Set", 399), ("Perfume Mist 100ml", 699),
    ],
    "Electronics Accessories": [
        ("20W Fast Charger", 699), ("Braided USB-C Cable 1.5m", 399), ("10000mAh Power Bank", 1299),
        ("Wireless Earbuds", 1799), ("Neckband Earphones", 999), ("Rugged Phone Case", 499), ("Tempered Glass Guard", 299),
        ("Smartwatch", 2499), ("Laptop Sleeve 14\"", 899), ("Bluetooth Speaker", 1999), ("Car Charger 38W", 549),
        ("Magnetic Phone Mount", 699), ("3-in-1 Wireless Charger", 2499), ("Gaming Mouse", 1199),
        ("Mechanical Keyboard", 3499), ("Ring Light 10\"", 999), ("Tripod Stand", 799), ("USB-C Hub 7-in-1", 1999),
        ("1080p Webcam", 2299), ("Smart LED Bulb", 499),
    ],
    "Home": [
        ("Handwoven Cotton Dhurrie", 1999), ("Brass Diya Set of 4", 799), ("Scented Soy Candle", 599),
        ("Ceramic Planter", 899), ("Block Print Bedsheet Set", 1899), ("Jute Cushion Cover", 499),
        ("Macrame Wall Hanging", 1299), ("Copper Water Bottle", 999), ("Terracotta Lamp", 1499), ("Rangoli Stencil Kit", 399),
        ("LED String Lights 10m", 449), ("Marigold Toran", 599), ("Wooden Serving Tray", 1199), ("Table Runner", 699),
        ("Pooja Thali Set", 1499), ("Ceramic Dinner Set", 3999), ("Cotton Bath Towel Set", 1299), ("Wall Clock", 1499),
        ("Floral Vase", 799), ("Aroma Diffuser", 1499),
    ],
    "Sweets": [
        ("Kaju Katli 500g", 799, True), ("Assorted Mithai Box 1kg", 1299, True), ("Dry Fruit Gift Hamper", 1499),
        ("Chocolate Truffle Box", 899), ("Mixed Namkeen Combo", 599), ("Gulkand Ladoo 500g", 649, True),
        ("Saffron Pista Barfi 500g", 849, True), ("Diwali Gift Hamper Deluxe", 2499), ("Roasted Almonds 250g", 449),
        ("Masala Cashews 200g", 399), ("Artisanal Peanut Chikki 250g", 249), ("Badam Pinni 500g", 699, True),
        ("Millet Cookies 300g", 299), ("Mango Pickle 400g", 349), ("Dark Chocolate Bar 100g", 249),
        ("Gourmet Gift Box", 1799), ("Pinni Ladoo 500g", 599, True), ("Ragi Cookies 300g", 279),
        ("Kesar Rasmalai Tin 1kg", 499), ("Thandai Mix 200g", 349),
    ],
}


@dataclass(frozen=True)
class MerchantType:
    platform: str
    label: str
    category: str                     # shown on the wall
    share: float                      # of the platform's merchants
    mix: tuple[tuple[str, int], ...]  # (product category, number of SKUs)
    d2c: bool = False
    footfall: float = 1.0             # relative order lines per day, before merchant size
    basket: float = 1.0               # order lines per customer visit


MERCHANT_TYPES = (
    MerchantType("shopify", "fashion", "Fashion", 0.30, (("Fashion", 22),), True, 1.2, 1.3),
    MerchantType("shopify", "beauty", "Beauty", 0.25, (("Beauty", 20),), True, 1.2, 1.6),
    MerchantType("shopify", "gadgets", "Electronics Accessories", 0.20, (("Electronics Accessories", 20),), True, 1.2, 1.3),
    MerchantType("shopify", "home", "Home", 0.17, (("Home", 20),), True, 1.0, 1.4),
    MerchantType("shopify", "gourmet", "Gourmet", 0.08, (("Sweets", 20),), True, 1.2, 1.8),
    MerchantType("reliance", "smart", "Grocery", 0.30, (("Staples", 10), ("Snacks", 7), ("Dairy", 4), ("Fresh", 4),
                                                         ("Personal Care", 4), ("Household", 4), ("Sweets", 3)), False, 4.0, 4.5),
    MerchantType("reliance", "digital", "Electronics", 0.20, (("Electronics", 18), ("Electronics Accessories", 10)), False, 2.0, 1.1),
    MerchantType("reliance", "trends", "Fashion", 0.20, (("Fashion", 22),), False, 1.5, 1.8),
    MerchantType("reliance", "jiomart", "Grocery", 0.30, (("Staples", 10), ("Snacks", 6), ("Personal Care", 4),
                                                          ("Household", 4)), False, 2.0, 3.0),
    MerchantType("dmart", "dmart", "Hypermarket", 1.0, (("Staples", 9), ("Snacks", 6), ("Dairy", 3), ("Fresh", 2),
                                                        ("Sweets", 3), ("Personal Care", 4), ("Household", 4),
                                                        ("Home", 3), ("Fashion", 3)), False, 7.0, 6.0),
)

BRAND_PREFIX = ["Urban", "Desi", "Rang", "Kesar", "Indigo", "Saffron", "Bombay", "Jaipur", "Khadi", "Mitti", "Neel",
                "Sutra", "Tana", "Noor", "Aura", "Veda", "Chai", "Jhumka", "Mehr", "Zari", "Kosha", "Ananta",
                "Pankh", "Suta", "Moha", "Tvam", "Ira", "Kaya", "Nila", "Baagh", "Gulmohar", "Chandni", "Ojas",
                "Prakriti", "Rasa", "Swadesh", "Tara", "Utsav", "Vayu", "Yatra"]
BRAND_SUFFIX = {
    "fashion": ["Threads", "Weaves", "Loom", "Studio", "Label", "Kurta Co.", "Drape", "Couture"],
    "beauty": ["Naturals", "Botanics", "Glow", "Ayurveda", "Skin Co.", "Essentials", "Beauty", "Organics"],
    "gadgets": ["Gear", "Tech", "Mobile", "Labs", "Circuit", "Charge", "Audio", "Wired"],
    "home": ["Living", "Home", "Decor", "Nest", "Ghar", "Interiors", "Crafts", "Abode"],
    "gourmet": ["Mithai", "Gourmet", "Treats", "Foods", "Kitchen", "Pantry", "Sweets", "Bites"],
}
SELLER_NAMES = ["Shree Ganesh", "Balaji", "Sai", "Laxmi", "Annapurna", "Krishna", "Om", "Jai Ambe", "Mahalaxmi",
                "Gurukripa", "Navkar", "Patel", "Sharma", "Agarwal", "Iyer", "Reddy", "Kumar", "Shiv Shakti",
                "Siddhi Vinayak", "Royal", "New India", "Janta", "Bharat", "Gupta", "Nair", "Joshi"]
SELLER_SUFFIX = ["Kirana", "Provisions", "Traders", "General Stores", "Supermart", "Enterprises", "Mart", "Bazaar"]
RELIANCE_STORE = {"smart": "Reliance Smart", "digital": "Reliance Digital", "trends": "Trends"}


@dataclass(frozen=True)
class Product:
    sku: str
    name: str
    category: str
    price: float
    cost: float
    lead_days: int
    perishable: bool


@dataclass(frozen=True)
class Merchant:
    id: str
    platform: str
    name: str
    city: str
    category: str
    kind: str      # MerchantType.label
    x: float
    y: float
    size: float    # relative footfall (hidden scale of demand)
    basket: float


@dataclass
class World:
    seed: int
    platforms: tuple[Platform, ...]
    merchants: list[Merchant]
    catalogs: list[list[Product]]
    # Customers, sorted by home merchant so each merchant's customers are one contiguous range.
    customer_merchant: np.ndarray   # int32 merchant index
    customer_interval: np.ndarray   # float32 days between purchases
    customer_lapse: np.ndarray      # float64 sim day (since SIM_START) they stop buying, inf if never
    pool_start: np.ndarray          # per merchant: first customer index
    pool_size: np.ndarray           # per merchant: number of customers

    @property
    def n_customers(self) -> int:
        return len(self.customer_merchant)

    def merchant_index(self) -> dict[str, int]:
        return {m.id: i for i, m in enumerate(self.merchants)}


def customer_id(index: int) -> str:
    return f"C{index:06d}"


def map_xy(lat: float, lon: float) -> tuple[float, float]:
    """The wall's projection of India: x = (lon - 68) / 29.5, y = (37.5 - lat) / 31 (y down), clamped to 0-1."""
    return (min(max((lon - 68) / 29.5, 0.0), 1.0), min(max((37.5 - lat) / 31, 0.0), 1.0))


def _split(total: int, shares: list[float]) -> list[int]:
    raw = np.array(shares) * total / sum(shares)
    counts = np.floor(raw).astype(int)
    for i in np.argsort(-(raw - counts))[: total - counts.sum()]:
        counts[i] += 1
    return counts.tolist()


def _round_price(p: float, d2c: bool) -> float:
    if p < 100:
        return float(max(5, round(p)))
    if d2c and p >= 300:
        return float(round(p / 50) * 50 - 1)
    return float(round(p))


def _names_d2c(rng: np.random.Generator, label: str, n: int, used: set[str]) -> list[str]:
    combos = [f"{p} {s}" for p in BRAND_PREFIX for s in BRAND_SUFFIX[label]]
    rng.shuffle(combos)
    out = [c for c in combos if c not in used][:n]
    while len(out) < n:
        out.append(f"{combos[len(out) % len(combos)]} {len(out) // len(combos) + 1}")
    used.update(out)
    return out


def _pick_city(rng: np.random.Generator, cities: list[str]) -> str:
    w = np.array([CITIES[c][2] for c in cities], float)
    return cities[rng.choice(len(cities), p=w / w.sum())]


def _store_area(rng: np.random.Generator, city: str, taken: set[str], brand: str) -> str:
    areas = AREAS.get(city, GENERIC_AREAS)
    free = [a for a in areas if f"{brand} {a}" not in taken]
    name = f"{brand} {free[rng.integers(len(free))]}" if free else None
    n = 2
    while name is None or name in taken:
        name = f"{brand} {areas[rng.integers(len(areas))]} {n}"
        n += 1
    taken.add(name)
    return name


# The demo's own company (docs/INDIA_WORLD.md) is the first Shopify merchant on the wall.
FLAGSHIP = {"name": "Desi Living", "city": "Bengaluru", "category": "Home Decor", "kind": "home"}


def _merchants(rng: np.random.Generator, total: int) -> list[Merchant]:
    out: list[Merchant] = []
    used: set[str] = {FLAGSHIP["name"]}
    all_cities = list(CITIES)
    for platform, n_platform in zip(PLATFORM_SHARE, _split(total, list(PLATFORM_SHARE.values()))):
        types = [t for t in MERCHANT_TYPES if t.platform == platform]
        counter = 0
        for mtype, n in zip(types, _split(n_platform, [t.share for t in types])):
            names = _names_d2c(rng, mtype.label, n, used) if mtype.d2c else None
            for i in range(n):
                counter += 1
                if platform == "dmart":
                    city = _pick_city(rng, DMART_CITIES)
                    name = _store_area(rng, city, used, "DMart")
                elif mtype.label in RELIANCE_STORE:
                    city = _pick_city(rng, all_cities)
                    name = _store_area(rng, city, used, RELIANCE_STORE[mtype.label])
                elif mtype.label == "jiomart":
                    city = _pick_city(rng, all_cities)
                    name = f"{SELLER_NAMES[rng.integers(len(SELLER_NAMES))]} {SELLER_SUFFIX[rng.integers(len(SELLER_SUFFIX))]}"
                    name = name if name not in used else f"{name} {city}"
                    name = name if name not in used else f"{name} {counter}"
                    used.add(name)
                else:
                    city = _pick_city(rng, all_cities)
                    name = names[i]
                lat, lon, _ = CITIES[city]
                x, y = map_xy(lat + rng.normal(0, 0.03), lon + rng.normal(0, 0.03))
                out.append(Merchant(
                    id=f"{platform}-{counter:04d}", platform=platform, name=name, city=city,
                    category=mtype.category, kind=mtype.label, x=round(x, 4), y=round(y, 4),
                    size=float(rng.lognormal(0, 0.4)) * mtype.footfall, basket=mtype.basket))
    first = next((i for i, m in enumerate(out) if m.platform == "shopify"), None)
    if first is not None:  # a well-known brand: at least 1.5x a typical D2C store's footfall
        home = next(t for t in MERCHANT_TYPES if t.platform == "shopify" and t.label == FLAGSHIP["kind"])
        x, y = map_xy(*CITIES[FLAGSHIP["city"]][:2])
        out[first] = replace(out[first], name=FLAGSHIP["name"], city=FLAGSHIP["city"], category=FLAGSHIP["category"],
                             kind=FLAGSHIP["kind"], x=round(x, 4), y=round(y, 4),
                             size=max(out[first].size, 1.5 * home.footfall), basket=home.basket)
    return out


def _catalog(rng: np.random.Generator, m: Merchant) -> list[Product]:
    mtype = next(t for t in MERCHANT_TYPES if t.platform == m.platform and t.label == m.kind)
    price_index = {"dmart": 0.95, "reliance": 0.99}.get(m.platform, 1.0) * rng.lognormal(0, 0.03)
    if mtype.d2c:
        price_index = rng.lognormal(0, 0.15)
    products = []
    for cat_name, n in mtype.mix:
        cat = CATEGORIES[cat_name]
        source = D2C[cat_name] if mtype.d2c else RETAIL[cat_name]
        prefix = f"{cat.code}{'D' if mtype.d2c else ''}"
        picks = rng.choice(len(source), size=min(n + int(rng.integers(-2, 3)), len(source)), replace=False)
        for j in sorted(picks):
            item = source[j]
            price = _round_price(item[1] * price_index * rng.lognormal(0, 0.02), mtype.d2c)
            margin = rng.uniform(*(D2C_MARGIN if mtype.d2c else cat.margin))
            lead = int(rng.integers(cat.lead_days[0], cat.lead_days[1] + 1))
            perishable = cat.perishable or (len(item) > 2 and item[2])
            products.append(Product(
                sku=f"{prefix}{j:03d}", name=f"{m.name} {item[0]}" if mtype.d2c else item[0],
                category=cat_name, price=price, cost=round(price * (1 - margin), 2),
                lead_days=1 if perishable and lead > 2 else lead, perishable=bool(perishable)))
    return products


def _customers(rng: np.random.Generator, merchants: list[Merchant], catalogs: list[list[Product]], n: int):
    interval = np.array([np.mean([CATEGORIES[p.category].interval_days for p in cat]) for cat in catalogs])
    visits = np.array([m.size / m.basket for m in merchants]) * np.array([len(c) for c in catalogs])
    weight = visits * interval
    pool = np.array(_split(n, list(weight)))
    pool = np.maximum(pool, 1) if n >= len(merchants) else pool
    while pool.sum() > n:
        pool[np.argmax(pool)] -= 1
    merchant = np.repeat(np.arange(len(merchants), dtype=np.int32), pool)
    start = np.concatenate([[0], np.cumsum(pool)[:-1]]).astype(np.int64)
    days = interval[merchant] * rng.lognormal(-0.12, 0.5, len(merchant))
    lapse = np.full(len(merchant), np.inf)
    lapsing = rng.random(len(merchant)) < 0.12
    lapse[lapsing] = rng.uniform(-30, 45, lapsing.sum())
    return merchant, days.astype(np.float32), lapse, start, pool.astype(np.int64)


# Shoppi categories -> the demand categories the simulator knows (base rate, elasticity, festivals).
SHOPPI_CATEGORY = {
    "Namkeen": "Snacks", "Ready Meals": "Snacks", "Frozen": "Snacks", "Sweets": "Sweets", "Gifting": "Sweets",
    "Staples": "Staples", "Beverages": "Staples", "Dairy": "Dairy", "Fruits & Vegetables": "Fresh",
    "Household": "Household", "Clothing": "Fashion", "Outerwear": "Fashion", "Accessories": "Fashion",
    "Footwear": "Fashion",
}
# Relative footfall (hidden demand scale) and order lines per visit, by merchant.
SHOPPI_SIZE = {"haldirams": (6.0, 2.5), "bigbasket": (9.0, 6.0), "fabindia": (3.0, 1.4), "bata": (3.0, 1.2)}
SHOPPI_LEAD = {"haldirams": 3, "bigbasket": 2, "fabindia": 14, "bata": 10}


def _read_csv(name: str) -> list[dict]:
    with open(SHOPPI_DIR / name, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _shoppi(rng: np.random.Generator) -> tuple[list[Merchant], list[list[Product]]]:
    """The Shoppi merchants and their real catalogs (examples/shoppi), one dot per merchant."""
    products = _read_csv("products.csv")
    merchants, catalogs = [], []
    for row in _read_csv("merchants.csv"):
        slug = row["slug"]
        size, basket = SHOPPI_SIZE.get(slug, (2.0, 1.5))
        mine = [p for p in products if p["merchant_id"] == row["id"]]
        cats = sorted({p["category"] for p in mine}, key=lambda c: -sum(p["category"] == c for p in mine))
        x, y = map_xy(float(row["lat"]), float(row["lng"]))
        merchants.append(Merchant(id=f"shoppi-{slug}", platform="shoppi", name=row["name"], city=row["city"],
                                  category=" & ".join(cats[:2]), kind=slug, x=round(x, 4), y=round(y, 4),
                                  size=size, basket=basket))
        catalog = []
        for p in mine:
            cat = SHOPPI_CATEGORY.get(p["category"], "Home")
            perishable = CATEGORIES[cat].perishable
            catalog.append(Product(sku=p["sku"], name=p["title"], category=cat, price=float(p["price"]),
                                   cost=float(p["cost"]), lead_days=1 if perishable else SHOPPI_LEAD.get(slug, 7),
                                   perishable=perishable))
        catalogs.append(catalog)
    return merchants, catalogs


def build_world(merchants: int = 600, customers: int = 120000, seed: int = 7) -> World:
    rng = np.random.default_rng(seed)
    if WORLD == "shoppi":
        ms, catalogs = _shoppi(rng)
        cm, ci, cl, start, size = _customers(rng, ms, catalogs, min(customers, 40000))
        return World(seed, PLATFORMS, ms, catalogs, cm, ci, cl, start, size)
    ms = _merchants(rng, merchants)
    catalogs = [_catalog(rng, m) for m in ms]
    cm, ci, cl, start, size = _customers(rng, ms, catalogs, customers)
    return World(seed, PLATFORMS, ms, catalogs, cm, ci, cl, start, size)


def world_payload(world: World, started_ms: int) -> dict:
    return {
        "started_ms": started_ms,
        "sim_start": contract.SIM_START,
        "platforms": [{"id": p.id, "name": p.name, "kind": p.kind, "color": p.color} for p in world.platforms],
        "merchants": [{"id": m.id, "platform": m.platform, "name": m.name, "city": m.city,
                       "category": m.category, "x": m.x, "y": m.y} for m in world.merchants],
        "customers": world.n_customers,
    }


def catalog_payload(products: list[Product], stock: list[int]) -> dict:
    return {"products": [{"sku": p.sku, "name": p.name, "category": p.category, "price": p.price, "cost": p.cost,
                          "stock": int(s), "lead_days": p.lead_days, "perishable": p.perishable}
                         for p, s in zip(products, stock)]}

