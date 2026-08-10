CATEGORIES = {
    "makanan_minuman": {
        "merchants": [
            "Warung Bu Sri", "McDonalds", "Starbucks", "Indomaret", "GoFood",
            "Kedai Kopi Senja", "KFC", "Pizza Hut", "Alfamart", "Sushi Tei",
            "Kopi Kenangan", "Solaria", "Bakmi GM", "HokBen", "Chatime",
        ],
        "held_out_test": ["Es Teh Indonesia", "Janji Jiwa", "Richeese Factory"],
        "amount_range": (15_000, 250_000),
    },
    "transportasi": {
        "merchants": ["Gojek", "Grab", "Pertamina", "MRT Jakarta", "Parkir Mall", "Shell", "Blue Bird", "TransJakarta"],
        "held_out_test": ["Vexcar Rental", "Whoosh Kereta Cepat"],
        "amount_range": (10_000, 500_000),
    },
    "belanja": {
        "merchants": ["Tokopedia", "Shopee", "Uniqlo", "Hypermart", "Zara", "H&M", "Lazada", "IKEA"],
        "held_out_test": ["Blibli", "MatahariMall"],
        "amount_range": (50_000, 2_000_000),
    },
    "tagihan_utilitas": {
        "merchants": ["PLN", "PDAM", "Indihome", "Telkomsel", "BPJS Kesehatan", "XL Axiata", "First Media"],
        "held_out_test": ["Biznet Home", "MyRepublic"],
        "amount_range": (100_000, 800_000),
    },
    "hiburan": {
        "merchants": ["Netflix", "Spotify", "CGV Cinema", "Steam", "PlayStation Store", "Disney Hotstar", "XXI Cinema"],
        "held_out_test": ["Vidio Premier", "WeTV"],
        "amount_range": (25_000, 500_000),
    },
    "kesehatan": {
        "merchants": ["Apotek K24", "RS Siloam", "Guardian", "Klinik Pratama", "Halodoc", "RS Pondok Indah", "Kimia Farma"],
        "held_out_test": ["Alodokter", "RS Mitra Keluarga"],
        "amount_range": (30_000, 3_000_000),
    },
    "pendidikan": {
        "merchants": ["Udemy", "Ruangguru", "Gramedia", "SPP Sekolah", "Coursera", "Zenius", "Skill Academy"],
        "held_out_test": ["Cakap", "Quipper"],
        "amount_range": (50_000, 5_000_000),
    },
    "investasi": {
        "merchants": ["Bibit", "Ajaib", "Bareksa", "Pluang", "Stockbit", "IPOT", "BIONS Sekuritas"],
        "held_out_test": ["Reksadana Syariah Mandiri", "Motion Trade"],
        "amount_range": (100_000, 10_000_000),
    },
    "transfer": {
        "merchants": ["Transfer ke Keluarga", "Transfer Teman", "Split Bill", "Transfer BCA", "Transfer Mandiri"],
        "held_out_test": ["Kirim Uang OVO", "Transfer Dana Darurat"],
        "amount_range": (50_000, 5_000_000),
    },
    "gaji_pendapatan": {
        "merchants": ["Gaji Bulanan", "Bonus", "Freelance Payment", "THR", "Komisi Penjualan"],
        "held_out_test": ["Dividen Saham", "Refund Pembelian"],
        "amount_range": (3_000_000, 25_000_000),
    },
    "asuransi": {
        "merchants": ["Prudential", "Allianz", "BPJS Ketenagakerjaan", "AXA Mandiri", "Sinarmas MSIG"],
        "held_out_test": ["Manulife Indonesia", "FWD Insurance"],
        "amount_range": (150_000, 1_500_000),
    },
    "donasi": {
        "merchants": ["Kitabisa", "Donasi Masjid", "Baznas", "Rumah Zakat", "ACT Foundation"],
        "held_out_test": ["Dompet Dhuafa", "Donasi Gereja"],
        "amount_range": (10_000, 500_000),
    },
    "lainnya": {
        "merchants": ["ATM Withdrawal", "Biaya Admin Bank", "Pembayaran Lain", "Materai Digital", "Denda Keterlambatan"],
        "held_out_test": ["Biaya Transfer Antar Bank", "Potongan Pajak"],
        "amount_range": (5_000, 200_000),
    },
}

# Untuk clustering: tiap segmen punya bobot kecenderungan kategori berbeda
USER_SEGMENTS = {
    "hemat": {
        "categories_weight": {
            "makanan_minuman": 0.22, "transportasi": 0.15, "tagihan_utilitas": 0.15,
            "investasi": 0.18, "gaji_pendapatan": 0.08, "pendidikan": 0.07,
            "donasi": 0.05, "lainnya": 0.10,
        },
        "transactions_per_month": (15, 25),
    },
    "boros": {
        "categories_weight": {
            "belanja": 0.28, "hiburan": 0.20, "makanan_minuman": 0.20,
            "transportasi": 0.12, "gaji_pendapatan": 0.06, "transfer": 0.09,
            "kesehatan": 0.05,
        },
        "transactions_per_month": (35, 55),
    },
    "investor": {
        "categories_weight": {
            "investasi": 0.32, "tagihan_utilitas": 0.12, "makanan_minuman": 0.16,
            "asuransi": 0.10, "gaji_pendapatan": 0.10, "pendidikan": 0.08,
            "transfer": 0.07, "lainnya": 0.05,
        },
        "transactions_per_month": (10, 20),
    },
    "standar": {
        "categories_weight": {
            "makanan_minuman": 0.20, "transportasi": 0.16, "belanja": 0.13,
            "tagihan_utilitas": 0.13, "hiburan": 0.08, "kesehatan": 0.10,
            "gaji_pendapatan": 0.07, "transfer": 0.06, "donasi": 0.04, "pendidikan": 0.03,
        },
        "transactions_per_month": (20, 35),
    },
}