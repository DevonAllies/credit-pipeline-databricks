from pyspark.sql import functions as F

NULL_TOKENS = r"(?i)^\s*(nd|nan|na|n/a|none|null|unknown|-|\.)?\s*$"
DATE_FORMATS = ["yyyy-MM-dd", "dd/MM/yyyy", "dd-MM-yyyy", "dd MMMM yyyy", "dd/MM/yy", "d MMMM yyyy", "d/M/yyyy"]

def clean_id(df, c = "customer_id", prefix = "C", width = 5):
    """c00042, C00042, 42 and 'C 0042' all become C00042. No digits at all become null."""
    digits = F.regexp_replace(F.col(c), "[^0-9]", "")
    return df.withColumn(c, F.when(digits != "", F.concat(F.lit(prefix), F.lpad(digits, width, "0"))))

def to_int(df, *cols):
    for c in cols:
        df = df.withColumn(c, F.col(c).try_cast("int"))
    return df

def parse_date(df, c, lo = "2000-01-01", hi = "2026-12-31"):
    attempts = [F.to_date(F.expr(f"try_to_timestamp(trim(`{c}`), '{fmt}')")) for fmt in DATE_FORMATS]
    parsed = F.coalesce(*attempts)
    parsed = F.when(parsed > F.lit(hi), F.add_months(parsed, -1200)).otherwise(parsed)
    valid = F.when((parsed >= F.lit(lo)) & (parsed <= F.lit(hi)), parsed)
    return df.withColumn(f"{c}_raw", F.col(c)).withColumn(c, valid)

def parse_number(df, c):
    """'12', '12.5' -> number; blank, ND, NH or anything else that is not a number -> null."""
    t = F.trim(F.col(c))
    return df.withColumn(c, F.when(t.rlike(NULL_TOKENS), F.lit(None).cast("double")).otherwise(t.try_cast("double")))

def parse_money(df, c):
    """'R 7350', 'R7,350', '7350.00' -> 7350.0 (strips R, spaces and thousands commas)."""
    t = F.regexp_replace(F.trim(F.col(c)), r"[Rr,\s]", "")
    return df.withColumn(c, F.when(F.col(c).rlike(NULL_TOKENS), F.lit(None).cast("double")).otherwise(t.try_cast("double")))

def null_outside(df, c, lo, hi):
    col = F.col(c)
    return df.withColumn(c, F.when((col >= lo) & (col <= hi), col))

def map_values(df, c, mapping, keep_unmapped=True):
    key = F.lower(F.trim(F.col(c)))
    lookup = F.create_map(*[F.lit(x) for kv in mapping.items() for x in kv])
    mapped = F.try_element_at(lookup, key)
    fallback = F.trim(F.col(c)) if keep_unmapped else F.lit(None)
    return df.withColumn(c, F.coalesce(mapped, fallback))

def report_unmapped(df, c, mapping):
    key = F.lower(F.trim(F.col(c)))
    return (df.filter(F.col(c).isNotNull() & ~key.isin(list(mapping)))
              .groupBy(c).count().orderBy(F.desc("count")))

SEX = {"m": "Male", "male": "Male", "f": "Female", "female": "Female"}
DECISION = {"approved": "Approved", "appr": "Approved", "declined": "Declined", "decl": "Declined"}
PROVINCE = {
    "gauteng": "Gauteng", "gau": "Gauteng",
    "kwazulu-natal": "KwaZulu-Natal", "kwa": "KwaZulu-Natal",
    "western cape": "Western Cape", "westerncape": "Western Cape", "wes": "Western Cape",
    "eastern cape": "Eastern Cape", "easterncape": "Eastern Cape", "eas": "Eastern Cape",
    "limpopo": "Limpopo", "lim": "Limpopo",
    "mpumalanga": "Mpumalanga", "mpu": "Mpumalanga",
    "north west": "North West", "northwest": "North West",
    "free state": "Free State", "freestate": "Free State", "fre": "Free State",
    "northern cape": "Northern Cape", "northerncape": "Northern Cape"
}

def test_partner_id_variants(spark):
    df = spark.createDataFrame([("p06",), (" P06",), ("P6",), ("P06 ",), ("TBC",)], ["partner_id"])
    out = [r.partner_id for r in clean_id(df, "partner_id", prefix="P", width=2).collect()]
    assert out == ["P06", "P06", "P06", "P06", None]

def test_campaign_spend_formats(spark):
    df = spark.createDataFrame([("R 1 113",), ("R3,564",), ("5,026.33",), ("2974.44",), ("TBC",), ("-31231.27",)], ["x"])
    out = [r.x for r in parse_money(df, "x").collect()]
    assert out == [1113.0, 3564.0, 5026.33, 2974.44, None, -31231.27]