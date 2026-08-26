import math
from collections import deque

from opendbc.can import CANParser
from opendbc.car import Bus, structs
from opendbc.car.common.conversions import Conversions as CV
from opendbc.car.interfaces import RadarInterfaceBase
from opendbc.car.hyundai.values import DBC, HyundaiFlags

RADAR_START_ADDR = 0x500
RADAR_MSG_COUNT = 32

# The Custin carries a 30 slot track list at 0x238 on bus 1 rather than the Mando 0x500 block.
# Fields were reverse engineered from this car's own logs, see opendbc/dbc/custin_radar.dbc.
CUSTIN_RADAR_START_ADDR = 0x238
CUSTIN_RADAR_MSG_COUNT = 30
CUSTIN_PRIMARY_ONLY = True  # see the comment in _update
CUSTIN_MIN_RANGE = 2.0      # below this is bumper clutter, and an empty slot reads zero
CUSTIN_MAX_ABS_Y = 5.5      # this lane and the two beside it; roadside structure sits outside
CUSTIN_WINDOW = 12          # ~0.36 s of history at 33 Hz
CUSTIN_MIN_SAMPLES = 6
CUSTIN_MAX_GAP = 0.2        # seconds without an update before a slot is considered reused
CUSTIN_MIN_SCORE = 30       # the radar's own tracking score, 31 is saturated
CUSTIN_MIN_HITS = 8         # consecutive good frames before a track is handed over
CUSTIN_MAX_JUMP = 3.0       # metres of range jump that means a different object took the slot


class CustinSlot:
  """The track list carries no speed field, so range is differentiated over a short window."""

  def __init__(self):
    self.hist: deque = deque(maxlen=CUSTIN_WINDOW)

  def reset(self):
    self.hist.clear()

  def update(self, t, d_rel):
    if self.hist and (t - self.hist[-1][0] > CUSTIN_MAX_GAP or abs(d_rel - self.hist[-1][1]) > CUSTIN_MAX_JUMP):
      self.hist.clear()
    self.hist.append((t, d_rel))

  def solve(self):
    """Least squares fit over the window, returning (dRel, vRel) or None.

    A range that does not move is a target keeping station with us, which happens for most
    of a traffic jam, so it still counts as valid and just gets a zero relative speed.
    """
    if not self.hist:
      return None
    if len(self.hist) < CUSTIN_MIN_SAMPLES:
      return self.hist[-1][1], 0.
    ts = [p[0] for p in self.hist]
    ds = [p[1] for p in self.hist]
    n = len(ts)
    mean_t = sum(ts) / n
    mean_d = sum(ds) / n
    var_t = sum((t - mean_t) ** 2 for t in ts)
    if var_t <= 0:
      return None
    v_rel = sum((t - mean_t) * (d - mean_d) for t, d in zip(ts, ds, strict=True)) / var_t
    return mean_d + v_rel * (ts[-1] - mean_t), v_rel

# POC for parsing corner radars: https://github.com/commaai/openpilot/pull/24221/


def get_radar_can_parser(CP):
  if Bus.radar not in DBC[CP.carFingerprint]:
    return None

  if CP.flags & HyundaiFlags.CUSTIN_RADAR:
    start_addr, msg_count, freq = CUSTIN_RADAR_START_ADDR, CUSTIN_RADAR_MSG_COUNT, 33
  else:
    start_addr, msg_count, freq = RADAR_START_ADDR, RADAR_MSG_COUNT, 50

  messages = [(f"RADAR_TRACK_{addr:x}", freq) for addr in range(start_addr, start_addr + msg_count)]
  return CANParser(DBC[CP.carFingerprint][Bus.radar], messages, 1)


def get_speed_can_parser(CP):
  # The Custin track list only carries each target's absolute speed, so the car's own
  # speed has to come off the powertrain bus to turn that into a relative speed.
  return CANParser(DBC[CP.carFingerprint][Bus.pt], [("WHL_SPD11", 50)], 0)


class RadarInterface(RadarInterfaceBase):
  def __init__(self, CP):
    super().__init__(CP)
    self.updated_messages = set()
    self.custin = bool(CP.flags & HyundaiFlags.CUSTIN_RADAR)
    self.start_addr = CUSTIN_RADAR_START_ADDR if self.custin else RADAR_START_ADDR
    self.msg_count = CUSTIN_RADAR_MSG_COUNT if self.custin else RADAR_MSG_COUNT
    self.trigger_msg = self.start_addr + self.msg_count - 1

    self.radar_off_can = CP.radarUnavailable
    self.rcp = get_radar_can_parser(CP)
    self.scp = get_speed_can_parser(CP) if self.custin and self.rcp is not None else None
    self.v_ego = 0.
    self.slots = {addr: CustinSlot() for addr in
                  range(CUSTIN_RADAR_START_ADDR, CUSTIN_RADAR_START_ADDR + CUSTIN_RADAR_MSG_COUNT)} if self.custin else {}
    self.frame_time = 0.
    self.hits: dict[int, int] = dict.fromkeys(self.slots, 0)

  def update(self, can_strings):
    if self.radar_off_can or (self.rcp is None):
      return super().update(None)

    vls = self.rcp.update(can_strings)
    self.updated_messages.update(vls)

    if self.custin and can_strings:
      self.frame_time = can_strings[-1][0] * 1e-9

    if self.scp is not None:
      self.scp.update(can_strings)
      whl = self.scp.vl["WHL_SPD11"]
      speeds = (whl["WHL_SPD_FL"], whl["WHL_SPD_FR"], whl["WHL_SPD_RL"], whl["WHL_SPD_RR"])
      self.v_ego = sum(speeds) / 4. * CV.KPH_TO_MS * self.CP.wheelSpeedFactor

    if self.trigger_msg not in self.updated_messages:
      return None

    rr = self._update(self.updated_messages)
    self.updated_messages.clear()

    return rr

  def _update(self, updated_messages):
    ret = structs.RadarData()
    if self.rcp is None:
      return ret

    if not self.rcp.can_valid:
      ret.errors.canError = True

    # 0x238 is the target the car's own ACC has locked onto, and it tracks the stock
    # ACC's reported distance to 0.10 m. The other 29 slots are the raw track list, where
    # a guardrail scanned along its length holds a steady range at a lane's worth of
    # lateral offset: radard reads that as a car keeping station ahead and picks it over
    # the real one, which measured worse than vision alone. Until the two can be told
    # apart, hand radard only the primary target.
    addrs = (self.start_addr,) if (self.custin and CUSTIN_PRIMARY_ONLY) else             range(self.start_addr, self.start_addr + self.msg_count)
    for addr in addrs:
      msg = self.rcp.vl[f"RADAR_TRACK_{addr:x}"]

      if addr not in self.pts:
        self.pts[addr] = structs.RadarData.RadarPoint()
        self.pts[addr].trackId = self.track_id
        self.track_id += 1

      if self.custin:
        rng = msg['LONG_DIST']
        azimuth = math.radians(msg['AZIMUTH'])
        y_rel = -math.sin(azimuth) * rng
        slot = self.slots[addr]

        # STATE does not tell targets from scenery here, and a guardrail read along its
        # length looks like a car at our own speed, so gate on range and lateral offset.
        keep = rng > CUSTIN_MIN_RANGE and msg['SCORE'] >= CUSTIN_MIN_SCORE and                (CUSTIN_PRIMARY_ONLY or abs(y_rel) < CUSTIN_MAX_ABS_Y)
        if keep:
          slot.update(self.frame_time, math.cos(azimuth) * rng)
          self.hits[addr] = min(self.hits[addr] + 1, CUSTIN_MIN_HITS)
        else:
          slot.reset()
          self.hits[addr] = 0

        fit = slot.solve()
        valid = fit is not None and self.hits[addr] >= CUSTIN_MIN_HITS
        if valid:
          self.pts[addr].dRel, self.pts[addr].vRel = fit
          self.pts[addr].yRel = y_rel
      else:
        valid = msg['STATE'] in (3, 4)
        if valid:
          azimuth = math.radians(msg['AZIMUTH'])
          self.pts[addr].dRel = math.cos(azimuth) * msg['LONG_DIST']
          self.pts[addr].yRel = 0.5 * -math.sin(azimuth) * msg['LONG_DIST']
          self.pts[addr].vRel = msg['REL_SPEED']

      if not valid:
        del self.pts[addr]

    ret.points = list(self.pts.values())
    return ret
