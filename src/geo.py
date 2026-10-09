from __future__ import annotations

import pandas as pd

# Compact mapping covers the dataset's most common country names; unmapped values remain 'Other'.
# This is used only for aggregate visual comparison, not geocoding.
CONTINENT_GROUPS = {
    'North America': ['United States of America','Canada','Mexico','Guatemala','Belize','Honduras','El Salvador','Nicaragua','Costa Rica','Panama','Cuba','Jamaica','Haiti','Dominican Republic','Bahamas','Barbados','Trinidad and Tobago'],
    'South America': ['Brazil','Argentina','Chile','Peru','Colombia','Venezuela','Ecuador','Bolivia','Paraguay','Uruguay','Guyana','Suriname'],
    'Europe': ['United Kingdom','France','Germany','Spain','Portugal','Italy','Ireland','Netherlands','Belgium','Switzerland','Austria','Poland','Czech Republic','Slovakia','Hungary','Romania','Bulgaria','Greece','Norway','Sweden','Finland','Denmark','Iceland','Ukraine','Croatia','Serbia','Slovenia','Bosnia and Herzegovina','Albania','North Macedonia','Estonia','Latvia','Lithuania','Moldova','Malta','Luxembourg','Cyprus'],
    'Asia': ['China','Japan','India','Pakistan','Bangladesh','Nepal','Bhutan','Sri Lanka','Maldives','Singapore','Malaysia','Indonesia','Philippines','Thailand','Vietnam','Cambodia','Laos','Myanmar','South Korea','North Korea','Mongolia','Afghanistan','Kazakhstan','Uzbekistan','Turkmenistan','Kyrgyzstan','Tajikistan','Iran','Iraq','Israel','Jordan','Lebanon','Saudi Arabia','United Arab Emirates','Qatar','Kuwait','Bahrain','Oman','Yemen','Turkey','Georgia','Armenia','Azerbaijan'],
    'Africa': ['Egypt','South Africa','Nigeria','Ghana','Kenya','Ethiopia','Morocco','Algeria','Tunisia','Libya','Sudan','South Sudan','Uganda','Tanzania','Rwanda','Burundi','Somalia','Senegal','Mali','Niger','Chad','Cameroon','Gabon','Congo','Democratic Republic of Congo','Angola','Zambia','Zimbabwe','Botswana','Namibia','Mozambique','Madagascar','Mauritius','Seychelles','Malawi','Sierra Leone','Liberia','Guinea','Guinea-Bissau','Gambia','Benin','Togo','Burkina Faso','Central African Republic','Equatorial Guinea','Eritrea','Djibouti','Lesotho','Eswatini'],
    'Oceania': ['Australia','New Zealand','Fiji','Papua New Guinea','Samoa','Tonga','Vanuatu','Solomon Islands','Kiribati','Micronesia','Palau','Marshall Islands','Nauru','Tuvalu']
}
COUNTRY_TO_CONTINENT = {country: continent for continent, countries in CONTINENT_GROUPS.items() for country in countries}


def add_continent(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out['continent'] = out['country'].map(COUNTRY_TO_CONTINENT).fillna('Other')
    return out
