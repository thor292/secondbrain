# The demo world (shared spec for the move to Indian commerce)

The main demo (Chat, Explore, Graph, HQ) and Live Commerce tell one story: SecondBrain is
connected to Shopify, Reliance Retail (JioMart) and DMart, and serves the merchants that sell
on them. Money is rupees, Indian digit grouping (₹12,34,567; ₹4.2 lakh; ₹1.3 crore; use
`secondbrain.money` in Python). Dates: today is Friday 6 November 2026; Dhanteras is today,
Diwali is Sunday 8 November 2026. Indian fiscal year April-March (FY2026-27 = Apr 2026-Mar 2027;
Q2 FY27 = Jul-Sep 2026).

## Company 1 (replaces Lumen Home): Desi Living

- id `desiliving`, name "Desi Living", sharing "patterns".
- Bengaluru-based D2C home decor and lifestyle brand, founded 2019. Handmade Indian home goods:
  brass, block print, dhurries, sheesham wood, Khurja ceramics.
- Channels: own Shopify store desiliving.in (~55% of revenue), Reliance JioMart marketplace
  (~30%), shop-in-shop shelves in 40 DMart stores in Mumbai and Pune (~15%).
- Warehouses: Hoskote (Bengaluru, main) and Bhiwandi (Mumbai). Carriers: BharatFleet Logistics
  (large items, restricted contract), Delhivery and Blue Dart (parcels), India Post for remote pincodes.
- Revenue: FY2025-26 ₹48.3 crore, growing ~20% a year (FY27 forecast ₹58 crore). Q2 FY27 (Jul-Sep 2026)
  ₹11.9 crore, up 21% from ₹9.9 crore; Q3 FY27 forecast ₹19.5 crore.
- Departments (ids unchanged): support "Customer Support", marketing "Marketing", operations
  "Operations & Fulfillment", finance "Finance", people "People & HR", merchandising "Merchandising".
- People (id: name, department, role, extra grants):
  - meera: Meera Iyer (CEO), executive
  - arjun: Arjun Mehta (Head of Support), support, lead
  - neha: Neha Kulkarni (Support agent), support, member, + operations internal
  - rohan: Rohan Kapoor (Marketing lead), marketing, lead, + merchandising internal
  - sana: Sana Sheikh (Content marketer), marketing, member
  - vikram: Vikram Rao (Operations manager), operations, lead
  - priya: Priya Nair (Warehouse coordinator), operations, member
  - anand: Anand Krishnan (CFO), finance, executive
  - tanvi: Tanvi Shah (Finance analyst), finance, member
  - farah: Farah Khan (People partner), people, lead
  - kabir: Kabir Singh (Merchandiser), merchandising, member
- Products (sku, name, category, price ₹, typical units a day at the start of the history in Sep 2024):
  - sheesham-sofa, Sheesham & linen 3-seater sofa, furniture, 48900, 5
  - brass-floor-lamp, Moradabad brass floor lamp, lighting, 7490, 34
  - dhurrie-rug-5x8, Handwoven cotton dhurrie 5x8, textiles, 6990, 26
  - kantha-throw, Kantha cotton throw, textiles, 1890, 105
  - blockprint-bedsheet, Jaipur block-print bedsheet set, textiles, 3490, 38
  - ceramic-lamp, Khurja ceramic table lamp, lighting, 2790, 28
  - sheesham-side-table, Sheesham wood side table, furniture, 8990, 9
  - diya-candle-set, Brass diya & sandalwood candle set, decor, 899, 70
- Suppliers: Moradabad brass workshop "Noor Metalcraft" (lamps; the damaged-shipment stories),
  Jaipur printers "Sanganer Prints", Panipat weavers "Panipat Looms", Saharanpur woodworkers
  "Saharanpur Woodcraft", Khurja potters "Khurja Ceramics Co.".
- Calendar that moves sales: Dhanteras and Diwali (the year's peak; gifting: diya sets, lamps,
  throws), the festive sale season from late September (Shopify/JioMart "Festive Sale"),
  Independence Day sale (15 Aug), Republic Day sale (26 Jan), wedding season (mid-Nov to mid-Dec,
  mid-Jan to Feb), Raksha Bandhan, Holi. Diwali shipping cutoff: orders by Dhanteras minus 2 days
  arrive before Diwali.
  Dates: Dhanteras 2024-10-29, Diwali 2024-11-01; Dhanteras 2025-10-18, Diwali 2025-10-20;
  Dhanteras 2026-11-06, Diwali 2026-11-08. Holi 2025-03-14, 2026-03-04. Raksha Bandhan 2024-08-19,
  2025-08-09, 2026-08-28. Festive sale starts 2024-09-27, 2025-09-22, 2026-10-03.
- Sales history in the database runs from 2024-09-01 to 2026-11-05 (yesterday).

## Company 2 (replaces Brightpath Logistics): BharatFleet Logistics

- id `bharatfleet`, name "BharatFleet Logistics", sharing "none" (its brain shares nothing upward:
  the demo of privacy). Pune-based large-item logistics partner; carries Desi Living's furniture.
- Department: operations "Operations". Person: karan: Karan Malhotra (Operations director), operations, lead.

## Live Commerce

Desi Living also appears as a merchant on the Live Commerce wall (Shopify, Bengaluru).
