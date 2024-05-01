from typing import Optional

import pandas as pd
from pandas.api.types import is_float_dtype, is_int64_dtype, is_object_dtype, is_string_dtype

from upgini.errors import ValidationError


class PhoneNormalizer:
    def __init__(self, df: pd.DataFrame, phone_column_name: str, country_column_name: Optional[str] = None):
        self.df = df
        self.phone_column_name = phone_column_name
        self.country_column_name = country_column_name

    def normalize(self) -> pd.DataFrame:
        self.phone_to_int()
        if self.country_column_name is not None:
            self.df = self.df.apply(self.add_prefix, axis=1)
        return self.df[self.phone_column_name].astype("Int64")

    def add_prefix(self, row):
        phone = row[self.phone_column_name]
        if pd.isna(phone):
            return row
        country = row[self.country_column_name]
        country_prefix_tuple = self.COUNTRIES_PREFIXES.get(country)
        if country_prefix_tuple is not None:
            country_prefix, number_of_digits = country_prefix_tuple
            if len(str(phone)) == number_of_digits:
                row[self.phone_column_name] = int(country_prefix + str(phone))
        return row

    def phone_to_int(self):
        """
        Convention: phone number is always presented as int number.
        phone_number = Country code + National Destination Code + Subscriber Number.
        Examples:
        41793834315     for Switzerland
        46767040672     for Sweden
        861065529988    for China
        18143008198     for the USA
        Inplace conversion of phone to int.

        Method will remove all non numeric chars from string and convert it to int.
        None will be set for phone numbers that couldn"t be converted to int
        """
        if is_string_dtype(self.df[self.phone_column_name]) or is_object_dtype(self.df[self.phone_column_name]):
            convert_func = self.phone_str_to_int_safe
        elif is_float_dtype(self.df[self.phone_column_name]):
            convert_func = self.phone_float_to_int_safe
        elif is_int64_dtype(self.df[self.phone_column_name]):
            convert_func = self.phone_int_to_int_safe
        else:
            raise ValidationError(
                f"phone_column_name {self.phone_column_name} doesn't have supported dtype. "
                f"Dataset dtypes: {self.df.dtypes}. "
                f"Contact developer and request to implement conversion of {self.phone_column_name} to int"
            )
        self.df[self.phone_column_name] = self.df[self.phone_column_name].apply(convert_func).astype("Int64")

    @staticmethod
    def phone_float_to_int_safe(value: float) -> Optional[int]:
        try:
            return PhoneNormalizer.validate_length(int(value))
        except Exception:
            return None

    @staticmethod
    def phone_int_to_int_safe(value: int) -> Optional[int]:
        try:
            return PhoneNormalizer.validate_length(int(value))
        except Exception:
            return None

    @staticmethod
    def phone_str_to_int_safe(value: str) -> Optional[int]:
        try:
            value = str(value)
            if value.endswith(".0"):
                value = value[: len(value) - 2]
            numeric_filter = filter(str.isdigit, value)
            numeric_string = "".join(numeric_filter)
            return PhoneNormalizer.validate_length(int(numeric_string))
        except Exception:
            return None

    @staticmethod
    def validate_length(value: int) -> Optional[int]:
        if value < 10000000 or value > 999999999999999:
            return None
        else:
            return value

    COUNTRIES_PREFIXES = {
        "US": ("1", 10),
        "CA": ("1", 10),
        "AI": ("1", 10),
        "AG": ("1", 10),
        "AS": ("1", 10),
        "BB": ("1", 10),
        "BS": ("1", 10),
        "VG": ("1", 10),
        "VI": ("1", 10),
        "KY": ("1", 10),
        "BM": ("1", 10),
        "GD": ("1", 10),
        "TC": ("1", 10),
        "MS": ("1", 10),
        "MP": ("1", 10),
        "GU": ("1", 10),
        "SX": ("1", 10),
        "LC": ("1", 10),
        "DM": ("1", 10),
        "VC": ("1", 10),
        "PR": ("1", 10),
        "TT": ("1", 10),
        "KN": ("1", 10),
        "JM": ("1", 10),
        "EG": ("20", 9),
        "SS": ("211", 9),
        "MA": ("212", 9),
        "EH": ("212", 4),
        "DZ": ("213", 8),
        "TN": ("216", 8),
        "LY": ("218", 9),
        "GM": ("220", 6),
        "SN": ("221", 9),
        "MR": ("222", 7),
        "ML": ("223", 8),
        "GN": ("224", 9),
        "CI": ("225", 7),
        "BF": ("226", 8),
        "NE": ("227", 8),
        "TG": ("228", 8),
        "BJ": ("229", 8),
        "MU": ("230", 7),
        "LR": ("231", 9),
        "SL": ("232", 8),
        "GH": ("233", 9),
        "NG": ("234", 9),
        "TD": ("235", 8),
        "CF": ("236", 7),
        "CM": ("237", 9),
        "CV": ("238", 7),
        "ST": ("239", 7),
        "GQ": ("240", 9),
        "GA": ("241", 8),
        "CG": ("242", 7),
        "CD": ("243", 9),
        "AO": ("244", 9),
        "GW": ("245", 6),
        "IO": ("246", 7),
        "AC": ("247", 5),
        "SC": ("248", 7),
        "SD": ("249", 9),
        "RW": ("250", 9),
        "ET": ("251", 9),
        "SO": ("252", 9),
        "DJ": ("253", 8),
        "KE": ("254", 9),
        "TZ": ("255", 9),
        "UG": ("256", 9),
        "BI": ("257", 8),
        "MZ": ("258", 8),
        "ZM": ("260", 9),
        "MG": ("261", 9),
        "RE": ("262", 9),
        "YT": ("262", 9),
        "TF": ("262", 9),
        "ZW": ("263", 9),
        "NA": ("264", 9),
        "MW": ("265", 7),
        "LS": ("266", 8),
        "BW": ("267", 7),
        "SZ": ("268", 8),
        "KM": ("269", 7),
        "ZA": ("27", 10),
        "SH": ("290", 5),
        "TA": ("290", 5),
        "ER": ("291", 7),
        "AT": ("43", 10),
        "AW": ("297", 7),
        "FO": ("298", 6),
        "GL": ("299", 6),
        "GR": ("30", 10),
        "BE": ("32", 8),
        "FR": ("33", 9),
        "ES": ("34", 9),
        "GI": ("350", 8),
        "PE": ("51", 8),
        "MX": ("52", 10),
        "CU": ("53", 8),
        "AR": ("54", 10),
        "BR": ("55", 10),
        "CL": ("56", 9),
        "CO": ("57", 8),
        "VE": ("58", 10),
        "PT": ("351", 9),
        "LU": ("352", 8),
        "IE": ("353", 8),
        "IS": ("354", 7),
        "AL": ("355", 8),
        "MT": ("356", 8),
        "CY": ("357", 8),
        "FI": ("358", 9),
        "BG": ("359", 8),
        "HU": ("36", 8),
        "LT": ("370", 8),
        "LV": ("371", 8),
        "EE": ("372", 7),
        "MD": ("373", 8),
        "AM": ("374", 8),
        "BY": ("375", 9),
        "AD": ("376", 6),
        "MC": ("377", 8),
        "SM": ("378", 9),
        "VA": ("3906698", 5),
        "UA": ("380", 9),
        "RS": ("381", 9),
        "ME": ("382", 8),
        "HR": ("385", 8),
        "SI": ("386", 8),
        "BA": ("387", 8),
        "MK": ("389", 8),
        "MY": ("60", 9),
        "AU": ("61", 9),
        "CX": ("61", 9),
        "CC": ("61", 9),
        "ID": ("62", 9),
        "PH": ("632", 7),
        "NZ": ("64", 8),
        "PN": ("64", 8),
        "SG": ("65", 8),
        "TH": ("66", 8),
        "IT": ("39", 10),
        "RO": ("40", 9),
        "CH": ("41", 9),
        "CZ": ("420", 9),
        "SK": ("421", 9),
        "GB": ("44", 10),
        "LI": ("423", 7),
        "GG": ("44", 10),
        "IM": ("44", 10),
        "JE": ("44", 10),
        "DK": ("45", 8),
        "SE": ("46", 8),
        "BD": ("880", 8),
        "TW": ("886", 9),
        "JP": ("81", 9),
        "KR": ("82", 9),
        "VN": ("84", 10),
        "KP": ("850", 8),
        "HK": ("852", 8),
        "MO": ("853", 8),
        "KH": ("855", 8),
        "LA": ("856", 8),
        "NO": ("47", 8),
        "SJ": ("47", 8),
        "BV": ("47", 8),
        "PL": ("48", 9),
        "DE": ("49", 10),
        "TR": ("90", 10),
        "IN": ("91", 10),
        "PK": ("92", 9),
        "AF": ("93", 9),
        "LK": ("94", 9),
        "MM": ("95", 7),
        "IR": ("98", 10),
        "MV": ("960", 7),
        "LB": ("961", 7),
        "JO": ("962", 9),
        "SY": ("963", 10),
        "IQ": ("964", 10),
        "KW": ("965", 7),
        "SA": ("966", 9),
        "YE": ("967", 7),
        "OM": ("968", 8),
        "PS": ("970", 8),
        "AE": ("971", 8),
        "IL": ("972", 9),
        "BH": ("973", 8),
        "QA": ("974", 8),
        "BT": ("975", 7),
        "MN": ("976", 8),
        "NP": ("977", 8),
        "TJ": ("992", 9),
        "TM": ("993", 8),
        "AZ": ("994", 9),
        "GE": ("995", 9),
        "KG": ("996", 9),
        "UZ": ("998", 9),
        "FK": ("500", 5),
        "BZ": ("501", 7),
        "GT": ("502", 8),
        "SV": ("503", 8),
        "HN": ("504", 8),
        "NI": ("505", 8),
        "CR": ("506", 8),
        "PA": ("507", 7),
        "PM": ("508", 6),
        "HT": ("509", 8),
        "GS": ("500", 5),
        "MF": ("590", 9),
        "BL": ("590", 9),
        "GP": ("590", 9),
        "BO": ("591", 9),
        "GY": ("592", 9),
        "EC": ("593", 9),
        "GF": ("594", 9),
        "PY": ("595", 9),
        "MQ": ("596", 9),
        "SR": ("597", 9),
        "UY": ("598", 9),
        "CW": ("599", 9),
        "BQ": ("599", 9),
        "RU": ("7", 10),
        "KZ": ("7", 10),
        "TL": ("670", 7),
        "NF": ("672", 7),
        "HM": ("672", 7),
        "BN": ("673", 7),
        "NR": ("674", 7),
        "PG": ("675", 7),
        "TO": ("676", 7),
        "SB": ("677", 7),
        "VU": ("678", 7),
        "FJ": ("679", 7),
        "PW": ("680", 7),
        "WF": ("681", 7),
        "CK": ("682", 5),
        "NU": ("683", 7),
        "WS": ("685", 7),
        "KI": ("686", 7),
        "NC": ("687", 7),
        "TV": ("688", 7),
        "PF": ("689", 7),
        "TK": ("690", 7),
        "FM": ("691", 7),
        "MH": ("692", 7),
    }
