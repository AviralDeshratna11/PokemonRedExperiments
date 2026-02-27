"""
pokemon_ram.py
--------------
Complete RAM address map and memory-reading utilities for Pokemon Red (GB).
All addresses verified against:
  - https://datacrystal.tcrf.net/wiki/Pok%C3%A9mon_Red_and_Blue/RAM_map
  - Bulbapedia Gen-I save data structure

Usage:
    from env.pokemon_ram import PokemonRAM
    ram = PokemonRAM(pyboy_instance)
    state = ram.read_all()
"""

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
#  RAW ADDRESS CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

class Addr:
    # ── Player position ───────────────────────────────────────────────────────
    MAP_ID          = 0xD35E   # current map number (0–247)
    Y_POS           = 0xD361   # player Y tile coordinate
    X_POS           = 0xD362   # player X tile coordinate
    PLAYER_DIR      = 0xC109   # 0=down 4=up 8=left 0x0C=right

    # ── Progress / Badges ─────────────────────────────────────────────────────
    BADGES          = 0xD356   # bitmask: bit0=Boulder…bit7=Earth

    # ── Party metadata ────────────────────────────────────────────────────────
    PARTY_COUNT     = 0xD163   # number of Pokemon in party (0–6)
    PARTY_SPECIES   = 0xD164   # species IDs [6 bytes]  D164–D169
    # Party struct starts at D16B; each mon is 44 bytes
    PARTY_BASE      = 0xD16B
    MON_STRUCT_SIZE = 44

    # Offsets within each mon struct (relative to PARTY_BASE + i*44)
    OFF_SPECIES      = 0x00
    OFF_CURRENT_HP_H = 0x01    # hi byte of current HP
    OFF_CURRENT_HP_L = 0x02    # lo byte
    OFF_STATUS       = 0x04    # status condition bitmask
    OFF_TYPE1        = 0x05
    OFF_TYPE2        = 0x06
    OFF_MOVE1        = 0x08
    OFF_MOVE2        = 0x09
    OFF_MOVE3        = 0x0A
    OFF_MOVE4        = 0x0B
    OFF_LEVEL_RAW    = 0x21    # "raw" level byte (not always actual level)
    OFF_MAX_HP_H     = 0x22    # hi byte of max HP
    OFF_MAX_HP_L     = 0x23    # lo byte
    OFF_ATTACK_H     = 0x24
    OFF_DEFENSE_H    = 0x26
    OFF_SPEED_H      = 0x28
    OFF_SPECIAL_H    = 0x2A
    OFF_LEVEL_ACTUAL = 0x21    # actual level (use D18C offset from base)

    # Actual levels are at D18C, D1B8, D1E4, D210, D23C, D268
    PARTY_LEVELS     = [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]
    # Max HP (2 bytes big-endian)
    PARTY_MAX_HP     = [0xD18D, 0xD1B9, 0xD1E5, 0xD211, 0xD23D, 0xD269]
    # Current HP (2 bytes big-endian) – at offset 0x01 from base
    PARTY_CUR_HP     = [0xD16C, 0xD198, 0xD1C4, 0xD1F0, 0xD21C, 0xD248]

    # ── Battle state ──────────────────────────────────────────────────────────
    IS_IN_BATTLE    = 0xD057   # 0=not in battle, 1=wild, 2=trainer
    BATTLE_TYPE     = 0xD05A   # battle type details
    ENEMY_LEVEL     = 0xCFE8   # current enemy pokemon level
    ENEMY_HP_H      = 0xCFE6   # enemy current HP hi byte
    ENEMY_HP_L      = 0xCFE7
    PLAYER_TURN     = 0xCCDC   # player selected move (0-indexed)
    IN_MENU         = 0xD057   # doubles as menu flag in some contexts

    # ── Money ─────────────────────────────────────────────────────────────────
    # BCD encoded: each nibble = one decimal digit
    MONEY_1         = 0xD347   # ten-thousands + hundred-thousands
    MONEY_2         = 0xD348   # tens + hundreds
    MONEY_3         = 0xD349   # units

    # ── Items ─────────────────────────────────────────────────────────────────
    BAG_COUNT       = 0xD31D   # number of item slots used
    BAG_START       = 0xD31E   # item bag: [id, qty] pairs, max 20 items (40 bytes)
    # Key items to check by item ID
    ITEM_HM_CUT     = 0xC4    # item ID for HM01 (Cut)
    ITEM_HM_FLASH   = 0xC5
    ITEM_HM_SURF    = 0xC9
    ITEM_HM_STRENGTH= 0xC7
    ITEM_HM_FLY     = 0xC6
    ITEM_BIKE       = 0x06
    ITEM_SILPH_SCOPE= 0x48
    ITEM_POKE_FLUTE = 0x49
    ITEM_OLD_ROD    = 0x4A
    ITEM_GOOD_ROD   = 0x4B
    ITEM_SUPER_ROD  = 0x4C

    # ── Event flags (key story progression) ───────────────────────────────────
    # Many are packed as bits within bytes across D7F1–D8CF
    # These specific addresses track individual story milestones
    EVENT_GOT_STARTER      = 0xD74B   # bit 1: received starter from Oak
    EVENT_OAKS_PARCEL       = 0xD74E   # bit 4: delivered Oak's parcel
    EVENT_GOT_POKEDEX      = 0xD74B   # bit 0: received Pokedex
    EVENT_BEAT_BROCK       = 0xD755   # bit 7 of D756... use badge check instead
    EVENT_BILLS_QUEST      = 0xD7F1   # bit 2: Bill's house event complete
    EVENT_BEAT_MISTY       = 0xD75E   # use badge check: bit 1 of D356
    EVENT_SS_ANNE_LEFT     = 0xD803   # bit 1: SS Anne has left port
    EVENT_BEAT_SURGE       = 0xD773   # bit 7: fought Lt. Surge
    EVENT_BEAT_ERIKA       = 0xD778   # bit 7
    EVENT_BEAT_KOGA        = 0xD77C   # bit 7
    EVENT_BEAT_SABRINA     = 0xD780   # bit 7
    EVENT_BEAT_BLAINE      = 0xD783   # bit 7
    EVENT_BEAT_GIOVANNI    = 0xD78C   # bit 7
    EVENT_SILPH_COMPLETE   = 0xD838   # bit 0: Silph Co defeated
    EVENT_GOT_LAPRAS       = 0xD854   # bit 0: received Lapras in Silph

    # ── Pokedex ───────────────────────────────────────────────────────────────
    POKEDEX_OWNED_START    = 0xD2F7   # 19 bytes, 1 bit per species
    POKEDEX_SEEN_START     = 0xD30A   # 19 bytes

    # ── Misc ──────────────────────────────────────────────────────────────────
    PLAYER_NAME_START      = 0xD158   # 11 bytes (text-encoded)
    CURRENT_BOX            = 0xDA80   # which PC box is active
    TOTAL_PLAY_FRAMES      = 0xDA44   # seconds played (lo byte)
    SAFARI_STEPS           = 0xD6D0   # 2 bytes: steps left in Safari Zone


# ─────────────────────────────────────────────────────────────────────────────
#  MAP SIZE TABLE (for exploration reward normalisation)
# ─────────────────────────────────────────────────────────────────────────────

MAP_SIZES = {
    # map_id: (width_tiles, height_tiles)  — approximate walkable area
    0x00: (20, 18),   # Pallet Town
    0x01: (20, 18),   # Viridian City
    0x02: (20, 18),   # Pewter City
    0x03: (20, 18),   # Cerulean City
    0x04: (30, 18),   # Lavender Town
    0x05: (40, 18),   # Vermilion City
    0x06: (30, 18),   # Celadon City
    0x07: (20, 18),   # Fuchsia City
    0x08: (40, 18),   # Cinnabar Island
    0x09: (20, 18),   # Saffron City
    0x0C: (80, 36),   # Route 1
    0x0D: (80, 36),   # Route 2
    0x0E: (160, 36),  # Route 3
    0x0F: (40, 36),   # Route 4
    0x10: (40, 36),   # Route 5
    0x11: (40, 36),   # Route 6
    0x12: (40, 36),   # Route 7
    0x13: (40, 36),   # Route 8
    0x14: (80, 36),   # Route 9
    0x15: (80, 36),   # Route 10
    0x16: (80, 36),   # Route 11
    0x17: (80, 36),   # Route 12
    0x18: (80, 36),   # Route 13
    0x19: (80, 36),   # Route 14
    0x1A: (80, 36),   # Route 15
    0x1B: (160, 36),  # Route 16
    0x1C: (160, 36),  # Route 17 (cycling road)
    0x1D: (80, 36),   # Route 18
    0x1E: (80, 54),   # Safari Zone entrance
    0x1F: (80, 54),   # Safari Zone East
    0x3C: (80, 36),   # Mt Moon (floor 1)
    0x3D: (80, 36),   # Mt Moon (floor 2)
    0x3E: (80, 36),   # Mt Moon (floor 3)
    0xE5: (80, 72),   # Victory Road
    0xF4: (80, 72),   # Silph Co
}

DEFAULT_MAP_SIZE = (40, 36)  # fallback for unknown maps


def get_map_size(map_id):
    w, h = MAP_SIZES.get(map_id, DEFAULT_MAP_SIZE)
    return w * h


# ─────────────────────────────────────────────────────────────────────────────
#  MAP ID → NAME
# ─────────────────────────────────────────────────────────────────────────────

MAP_NAMES = {
    0x00: "Pallet Town",
    0x01: "Viridian City",
    0x02: "Pewter City",
    0x03: "Cerulean City",
    0x04: "Lavender Town",
    0x05: "Vermilion City",
    0x06: "Celadon City",
    0x07: "Fuchsia City",
    0x08: "Cinnabar Island",
    0x09: "Saffron City",
    0x0A: "Unknown Dungeon (Cerulean Cave)",
    0x0B: "Route 22",
    0x0C: "Route 1",
    0x0D: "Route 2",
    0x0E: "Route 3",
    0x0F: "Route 4",
    0x10: "Route 5",
    0x11: "Route 6",
    0x12: "Route 7",
    0x13: "Route 8",
    0x14: "Route 9",
    0x15: "Route 10",
    0x16: "Route 11",
    0x17: "Route 12",
    0x18: "Route 13",
    0x19: "Route 14",
    0x1A: "Route 15",
    0x1B: "Route 16",
    0x1C: "Route 17",
    0x1D: "Route 18",
    0x1E: "Route 19",
    0x1F: "Route 20",
    0x20: "Route 21",
    0x21: "Route 23",
    0x22: "Route 24",
    0x23: "Route 25",
    0x3C: "Mt. Moon (1F)",
    0x3D: "Mt. Moon (B1F)",
    0x3E: "Mt. Moon (B2F)",
    0x59: "Viridian Gym",
    0x5C: "Pewter Gym",
    0x5D: "Cerulean Gym",
    0x5E: "Vermilion Gym",
    0x5F: "Celadon Gym",
    0x60: "Fuchsia Gym",
    0x61: "Saffron Gym",
    0x62: "Cinnabar Gym",
    0x99: "Oak's Lab",
    0xF4: "Silph Co (1F)",
    0xE5: "Victory Road (1F)",
    0xF0: "Indigo Plateau",
}


# ─────────────────────────────────────────────────────────────────────────────
#  POKEMON SPECIES ID → NAME (first 151 + key ones)
# ─────────────────────────────────────────────────────────────────────────────

SPECIES_NAMES = {
    0x99: "Bulbasaur", 0x09: "Ivysaur",  0x9A: "Venusaur",
    0xB0: "Charmander",0xB2: "Charmeleon",0xB4: "Charizard",
    0xB1: "Squirtle",  0xB3: "Wartortle",0x1C: "Blastoise",
    0x7B: "Caterpie",  0x7C: "Metapod",  0x7D: "Butterfree",
    0x6D: "Weedle",    0x6E: "Kakuna",   0x6F: "Beedrill",
    0x96: "Pidgey",    0x97: "Pidgeotto",0x98: "Pidgeot",
    0x1D: "Rattata",   0x1E: "Raticate",
    0x50: "Pikachu",   0x55: "Raichu",
    # ... (abbreviated for space; a full lookup is rarely needed at runtime)
}


# ─────────────────────────────────────────────────────────────────────────────
#  HELPER: decode BCD money value
# ─────────────────────────────────────────────────────────────────────────────

def decode_bcd(hi, mid, lo):
    """Decode three BCD bytes into an integer money value (max 999999)."""
    def nibbles(b):
        return (b >> 4) * 10 + (b & 0x0F)
    return nibbles(hi) * 10000 + nibbles(mid) * 100 + nibbles(lo)


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN RAM READER CLASS
# ─────────────────────────────────────────────────────────────────────────────

class PokemonRAM:
    """
    Wraps a PyBoy instance and provides structured access to all game state.

    Usage:
        pyboy = PyBoy("PokemonRed.gb", window_type="headless")
        ram   = PokemonRAM(pyboy)
        state = ram.read_all()          # full dict
        obs   = ram.to_feature_vector() # flat np.float32 for neural net
    """

    def __init__(self, pyboy):
        self.pyboy = pyboy

    # ─── Low-level helpers ────────────────────────────────────────────────────

    def _m(self, addr):
        """Read a single byte from RAM."""
        return self.pyboy.memory[addr]

    def _m16(self, addr_hi):
        """Read a big-endian 16-bit value (two consecutive bytes)."""
        return (self._m(addr_hi) << 8) | self._m(addr_hi + 1)

    def _bit(self, addr, bit):
        """Check a single bit (0=LSB) at a RAM address."""
        return bool((self._m(addr) >> bit) & 1)

    # ─── Position ─────────────────────────────────────────────────────────────

    def read_position(self):
        return {
            "map_id": self._m(Addr.MAP_ID),
            "x":      self._m(Addr.X_POS),
            "y":      self._m(Addr.Y_POS),
            "map_name": MAP_NAMES.get(self._m(Addr.MAP_ID), f"Map_{self._m(Addr.MAP_ID):02X}"),
        }

    # ─── Badges ───────────────────────────────────────────────────────────────

    def read_badges(self):
        raw = self._m(Addr.BADGES)
        names = ["Boulder","Cascade","Thunder","Rainbow","Soul","Marsh","Volcano","Earth"]
        return {
            "raw":   raw,
            "count": bin(raw).count("1"),
            "flags": {names[i]: bool((raw >> i) & 1) for i in range(8)},
        }

    # ─── Party ────────────────────────────────────────────────────────────────

    def read_party(self):
        count = min(self._m(Addr.PARTY_COUNT), 6)
        mons = []
        for i in range(count):
            base = Addr.PARTY_BASE + i * Addr.MON_STRUCT_SIZE
            cur_hp  = self._m16(Addr.PARTY_CUR_HP[i])
            max_hp  = self._m16(Addr.PARTY_MAX_HP[i])
            level   = self._m(Addr.PARTY_LEVELS[i])
            species = self._m(base + Addr.OFF_SPECIES)
            status  = self._m(base + Addr.OFF_STATUS)
            mons.append({
                "species":  species,
                "name":     SPECIES_NAMES.get(species, f"PKM_{species:02X}"),
                "level":    level,
                "current_hp": cur_hp,
                "max_hp":   max_hp,
                "hp_frac":  cur_hp / max_hp if max_hp > 0 else 0.0,
                "fainted":  cur_hp == 0,
                "status":   status,  # 0=none, 4=psn, 8=par, etc.
                "moves": [
                    self._m(base + Addr.OFF_MOVE1),
                    self._m(base + Addr.OFF_MOVE2),
                    self._m(base + Addr.OFF_MOVE3),
                    self._m(base + Addr.OFF_MOVE4),
                ],
            })
        return {
            "count": count,
            "mons":  mons,
            "all_fainted":    count > 0 and all(m["fainted"] for m in mons),
            "total_level":    sum(m["level"] for m in mons),
            "alive_count":    sum(1 for m in mons if not m["fainted"]),
            "avg_hp_frac":    np.mean([m["hp_frac"] for m in mons]) if mons else 0.0,
        }

    # ─── Battle ───────────────────────────────────────────────────────────────

    def read_battle(self):
        battle_val = self._m(Addr.IS_IN_BATTLE)
        return {
            "in_battle":     battle_val != 0,
            "is_wild":       battle_val == 1,
            "is_trainer":    battle_val == 2,
            "enemy_level":   self._m(Addr.ENEMY_LEVEL),
            "enemy_cur_hp":  self._m16(Addr.ENEMY_HP_H),
        }

    # ─── Money ────────────────────────────────────────────────────────────────

    def read_money(self):
        return decode_bcd(
            self._m(Addr.MONEY_1),
            self._m(Addr.MONEY_2),
            self._m(Addr.MONEY_3),
        )

    # ─── Items ────────────────────────────────────────────────────────────────

    def read_items(self):
        count = min(self._m(Addr.BAG_COUNT), 20)
        items = {}
        for i in range(count):
            item_id  = self._m(Addr.BAG_START + i * 2)
            item_qty = self._m(Addr.BAG_START + i * 2 + 1)
            items[item_id] = item_qty
        return {
            "raw":          items,
            "has_cut_hm":   Addr.ITEM_HM_CUT    in items,
            "has_surf_hm":  Addr.ITEM_HM_SURF   in items,
            "has_fly_hm":   Addr.ITEM_HM_FLY    in items,
            "has_flash_hm": Addr.ITEM_HM_FLASH  in items,
            "has_strength_hm": Addr.ITEM_HM_STRENGTH in items,
            "has_bike":     Addr.ITEM_BIKE       in items,
            "has_silph_scope": Addr.ITEM_SILPH_SCOPE in items,
            "has_poke_flute":  Addr.ITEM_POKE_FLUTE in items,
        }

    # ─── Events ───────────────────────────────────────────────────────────────

    def read_events(self):
        """Read key story event flags."""
        badges = self._m(Addr.BADGES)
        return {
            "got_starter":      self._bit(Addr.EVENT_GOT_STARTER, 1),
            "got_pokedex":      self._bit(Addr.EVENT_GOT_STARTER, 0),
            "delivered_parcel": self._bit(Addr.EVENT_OAKS_PARCEL, 4),
            "bills_quest_done": self._bit(Addr.EVENT_BILLS_QUEST, 2),
            "ss_anne_left":     self._bit(Addr.EVENT_SS_ANNE_LEFT, 1),
            "silph_complete":   self._bit(Addr.EVENT_SILPH_COMPLETE, 0),
            "got_lapras":       self._bit(Addr.EVENT_GOT_LAPRAS, 0),
            # Badge-derived events (more reliable than event bytes for gyms)
            "beat_brock":   bool((badges >> 0) & 1),
            "beat_misty":   bool((badges >> 1) & 1),
            "beat_surge":   bool((badges >> 2) & 1),
            "beat_erika":   bool((badges >> 3) & 1),
            "beat_koga":    bool((badges >> 4) & 1),
            "beat_sabrina": bool((badges >> 5) & 1),
            "beat_blaine":  bool((badges >> 6) & 1),
            "beat_giovanni":bool((badges >> 7) & 1),
        }

    # ─── Pokedex ──────────────────────────────────────────────────────────────

    def read_pokedex_counts(self):
        owned = sum(bin(self._m(Addr.POKEDEX_OWNED_START + i)).count("1") for i in range(19))
        seen  = sum(bin(self._m(Addr.POKEDEX_SEEN_START  + i)).count("1") for i in range(19))
        return {"owned": owned, "seen": seen}

    # ─── Full state dict ──────────────────────────────────────────────────────

    def read_all(self):
        """Return complete structured game state as a dict."""
        return {
            **self.read_position(),
            "badges":   self.read_badges(),
            "party":    self.read_party(),
            "battle":   self.read_battle(),
            "money":    self.read_money(),
            "items":    self.read_items(),
            "events":   self.read_events(),
            "pokedex":  self.read_pokedex_counts(),
        }

    # ─── Feature vector for neural network ────────────────────────────────────

    def to_feature_vector(self):
        """
        Returns a flat np.float32 array suitable as neural network input.
        Length: 128 features (documented below).
        """
        s = self.read_all()
        feats = []

        # [0-2] Position (normalised 0–1)
        feats.append(s["map_id"] / 247.0)
        feats.append(s["x"]      / 255.0)
        feats.append(s["y"]      / 255.0)

        # [3] Badge count (normalised)
        feats.append(s["badges"]["count"] / 8.0)

        # [4-11] Individual badge flags
        badge_names = ["Boulder","Cascade","Thunder","Rainbow","Soul","Marsh","Volcano","Earth"]
        for name in badge_names:
            feats.append(float(s["badges"]["flags"][name]))

        # [12-17] Party: HP fraction for each slot (0 if empty)
        for i in range(6):
            if i < s["party"]["count"]:
                feats.append(s["party"]["mons"][i]["hp_frac"])
            else:
                feats.append(0.0)

        # [18-23] Party: level / 100 for each slot
        for i in range(6):
            if i < s["party"]["count"]:
                feats.append(s["party"]["mons"][i]["level"] / 100.0)
            else:
                feats.append(0.0)

        # [24] Party: total level / 600
        feats.append(s["party"]["total_level"] / 600.0)

        # [25] Party: alive fraction
        feats.append(s["party"]["alive_count"] / 6.0)

        # [26] In battle flag
        feats.append(float(s["battle"]["in_battle"]))

        # [27] Wild vs trainer
        feats.append(float(s["battle"]["is_trainer"]))

        # [28] Money (log-normalised)
        import math
        feats.append(math.log1p(s["money"]) / math.log1p(999999))

        # [29-36] Key item flags
        items = s["items"]
        for key in ["has_cut_hm","has_surf_hm","has_fly_hm","has_flash_hm",
                    "has_strength_hm","has_bike","has_silph_scope","has_poke_flute"]:
            feats.append(float(items[key]))

        # [37-52] Event flags
        events = s["events"]
        for key in ["got_starter","got_pokedex","delivered_parcel","bills_quest_done",
                    "ss_anne_left","silph_complete","got_lapras",
                    "beat_brock","beat_misty","beat_surge","beat_erika",
                    "beat_koga","beat_sabrina","beat_blaine","beat_giovanni"]:
            feats.append(float(events[key]))

        # Pad to 128 features with zeros (reserved for future use)
        while len(feats) < 128:
            feats.append(0.0)

        return np.array(feats[:128], dtype=np.float32)
