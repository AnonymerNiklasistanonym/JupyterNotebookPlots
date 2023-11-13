from decimal import Decimal
from fractions import Fraction
import math
import pandas as pd


BACKSLASH = "\\"


def convert_float_to_latex_frac_str(num: float, dfrac=False) -> str:
    frac = Fraction(num).limit_denominator()
    return (
        f"{BACKSLASH}{'d' if dfrac else ''}frac"
        + "{"
        + f"{frac.numerator}"
        + "}{"
        + f"{frac.denominator}"
        + "}"
    )


def latex_float(num: float, num_decimal_places=4):
    float_str = "{0:.4g}".format(num)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"{0} \cdot 10^{{{1}}}".format(base, int(exponent))
    else:
        return round(num, num_decimal_places)


def convert_float_to_latex_frac_approx_str(
    num: float, dfrac=False, num_decimal_places=4
) -> str:
    # Check if the number is exact or should be rounded
    approx = Decimal(f"{num}").as_tuple().exponent < -num_decimal_places
    approx_close = math.isclose(num, round(num, num_decimal_places))
    # Check if a fraction can be created
    frac = Fraction(num).limit_denominator()
    create_frac = (
        len(f"{frac.denominator}") < 4
        and frac.numerator != 0
        and not (frac.numerator == 1 and frac.denominator == 1)
    )
    return f"{convert_float_to_latex_frac_str(num) + ' ' if create_frac else ''}{f'{BACKSLASH}approx ' if approx and not approx_close else '= ' if create_frac else ''}{latex_float(num, num_decimal_places) if approx or approx_close else num}"


def convert_pd_df_cols_float_to_latex_str(
    pd_df: pd.DataFrame, cols: list[str], num_decimal_places=4
) -> str:
    for col in cols:
        pd_df[col] = [
            f"${convert_float_to_latex_frac_approx_str(x, num_decimal_places)}$"
            for x in pd_df[col]
        ]
    return pd_df
