---
sort: 6
---
# Issue an Instruction
![Image](./figures/Issue.PNG)

## Barriers
CUDA supports different kinds of barriers like `membar` in PTX and other warp-wide or block-wide sync barriers. They are modeled by `shd_warp_t::set_membar()` and `shd_warp_t::clear_membar()`, or the `barrier_ste_t m_barriers` in the `shader_core_ctx`.

## Scoreboard
The scoreboard is used to avoid read-after-write or write-after-write data hazards. In Accel-SIM, the scoreboard has two lists to reserve the destination registers of issued instructions. The first one, `reg_table`, tracks all the destination registers. The second one, `longopregs`, only tracks the destination registers of memory access.

Once an instruction of a warp is issued, its destination registers are reserved in the scoreboard. The reserved regsiters will be released in the writeback stage of the SM core's pipeline.

An instruction cannot be issued if its source registers or destination registers are reserved in the scoreboard of its hardware warp.

> In the SASS code, there are 6 barriers can be assigned to avoid data hazard. For instance, if a preceding instruction is marked with barrier 1, and a following instruction is maked with wait barrier 0x000010b, then the next instruction cannot be issued before the previous one is finished. However, this kind of barrier is not modeled in the Accel-SIM. 

> We guess that as each warp is allowed to access a large volumn of registers, it is too costly to track all the conflicts in the hardware. Therefore, the compiler resolves the data dependency between instructions, and mark the unavoidable hazards with one of the 6 barriers. The hardware only has to check the 6 barriers, which could be much simpler.

The class `Scoreboard` is defined in `scoreboard.h` as follows
```c++
class Scoreboard {
 public:
  Scoreboard(unsigned sid, unsigned n_warps, class gpgpu_t *gpu);

  void reserveRegisters(const warp_inst_t *inst);
  void releaseRegisters(const warp_inst_t *inst);
  void releaseRegister(unsigned wid, unsigned regnum);

  bool checkCollision(unsigned wid, const inst_t *inst) const;
  bool pendingWrites(unsigned wid) const;
  void printContents() const;
  const bool islongop(unsigned warp_id, unsigned regnum);

 private:
  void reserveRegister(unsigned wid, unsigned regnum);
  int get_sid() const { return m_sid; }

  unsigned m_sid;

  // keeps track of pending writes to registers
  // indexed by warp id, reg_id => pending write count
  std::vector<std::set<unsigned> > reg_table;
  // Register that depend on a long operation (global, local or tex memory)
  std::vector<std::set<unsigned> > longopregs;

  class gpgpu_t *m_gpu;
};
```
The Scoreboard is initialized as follows
```c++
// shader_core_ctx::create_schedulers()
m_scoreboard = new Scoreboard(m_sid, m_config->max_warps_per_shader, m_gpu);

// sid: SM id
// n_warps: 2048 / 32
Scoreboard::Scoreboard(unsigned sid, unsigned n_warps, class gpgpu_t* gpu)
    : longopregs() {
  m_sid = sid;
  // Initialize size of table
  reg_table.resize(n_warps);
  longopregs.resize(n_warps);

  m_gpu = gpu;
}
```
So each warp has a vector of  `std::set<unsigned>` for `reg_table` and  a vector of  `std::set<unsigned>` for `longopregs`. The length of the vector equals to the number of hardware warps.
```c++
// wid: hw warp id
// regnum: the register to reserve

void Scoreboard::reserveRegister(unsigned wid, unsigned regnum) {
  if (!(reg_table[wid].find(regnum) == reg_table[wid].end())) {
    printf(
        "Error: trying to reserve an already reserved register (sid=%d, "
        "wid=%d, regnum=%d).",
        m_sid, wid, regnum);
    abort();
  }
  SHADER_DPRINTF(SCOREBOARD, "Reserved Register - warp:%d, reg: %d\n", wid,
                 regnum);
  reg_table[wid].insert(regnum);
  // It simply insert the regnum into the set.
}

void Scoreboard::releaseRegister(unsigned wid, unsigned regnum) {
  if (!(reg_table[wid].find(regnum) != reg_table[wid].end())) return;
  SHADER_DPRINTF(SCOREBOARD, "Release register - warp:%d, reg: %d\n", wid,
                 regnum);
  // remove regnum from the set
  reg_table[wid].erase(regnum);
}
```
At last, the `checkCollision`
```c++
bool Scoreboard::checkCollision(unsigned wid, const class inst_t* inst) const {
  // Get list of all input and output registers
  std::set<int> inst_regs;

  for (unsigned iii = 0; iii < inst->outcount; iii++)
    inst_regs.insert(inst->out[iii]);

  for (unsigned jjj = 0; jjj < inst->incount; jjj++)
    inst_regs.insert(inst->in[jjj]);

  if (inst->pred > 0) inst_regs.insert(inst->pred);
  if (inst->ar1 > 0) inst_regs.insert(inst->ar1);
  if (inst->ar2 > 0) inst_regs.insert(inst->ar2);

  // Check for collision, get the intersection of reserved registers and
  // instruction registers
  std::set<int>::const_iterator it2;
  // loop through all the input and output registers used by the instruction
  for (it2 = inst_regs.begin(); it2 != inst_regs.end(); it2++)
    // set::find: Searches the container for an element equivalent to val 
    // and returns an iterator to it if found, otherwise it returns an iterator to set::end.
    if (reg_table[wid].find(*it2) != reg_table[wid].end()) {
      return true;
    }
  return false;
}

// if the regtable for the warp is not empty, it is pending writes
bool Scoreboard::pendingWrites(unsigned wid) const {
  return !reg_table[wid].empty();
}
```
Let's check the longop.
```c++
// if the regnum is in the longopregs, it is longopreg
const bool Scoreboard::islongop(unsigned warp_id, unsigned regnum) {
  return longopregs[warp_id].find(regnum) != longopregs[warp_id].end();
}

// reserve registers with a warp_inst_t
void Scoreboard::reserveRegisters(const class warp_inst_t* inst) {
  // traverse all the output registers
  for (unsigned r = 0; r < MAX_OUTPUT_VALUES; r++) {
    // if the output register is valid
    if (inst->out[r] > 0) {
      // reserve the output register
      reserveRegister(inst->warp_id(), inst->out[r]);
      SHADER_DPRINTF(SCOREBOARD, "Reserved register - warp:%d, reg: %d\n",
                     inst->warp_id(), inst->out[r]);
    }
  }
	// if the operation is load
  // Keep track of long operations
  if (inst->is_load() && (inst->space.get_type() == global_space ||
                          inst->space.get_type() == local_space ||
                          inst->space.get_type() == param_space_kernel ||
                          inst->space.get_type() == param_space_local ||
                          inst->space.get_type() == param_space_unclassified ||
                          inst->space.get_type() == tex_space)) {
    // the output registers of load instruction are put into the longopregs
    for (unsigned r = 0; r < MAX_OUTPUT_VALUES; r++) {
      if (inst->out[r] > 0) {
        SHADER_DPRINTF(SCOREBOARD, "New longopreg marked - warp:%d, reg: %d\n",
                       inst->warp_id(), inst->out[r]);
        longopregs[inst->warp_id()].insert(inst->out[r]);
      }
    }
  }
}

void Scoreboard::releaseRegisters(const class warp_inst_t* inst) {
  // traverse all the output registers
  for (unsigned r = 0; r < MAX_OUTPUT_VALUES; r++) {
    if (inst->out[r] > 0) {
      SHADER_DPRINTF(SCOREBOARD, "Register Released - warp:%d, reg: %d\n",
                     inst->warp_id(), inst->out[r]);
      // release the output registers
      releaseRegister(inst->warp_id(), inst->out[r]);
      // release the registers in the longopregs
      longopregs[inst->warp_id()].erase(inst->out[r]);
    }
  }
}
```
A brief summary of scoreboard

* Each hardware warp has two `std::set`: `reg_table[warp_id]` and `longopregs[warp_id]`, one tracks the output registers, the other tracks the longops like memory access
* `void reserveRegisters(const warp_inst_t *inst)`: reserve all the output registers in the scoreboard `reg_table`. If the instruction is memory read (load), the output registers are also reserved in the `longopregs`
* `void releaseRegisters(const warp_inst_t *inst)`: release all the output regs of the instruction from both `reg_table` and `longopregs`
* `bool checkCollision(unsigned wid, const inst_t *inst) const`: as long as one of the input/output registers in the instruction is in the `reg_table`, it returns true.
* `bool pendingWrites(unsigned wid) const`: the warp is pendingWrite as long as its score board is not empty.
* `const bool islongop(unsigned warp_id, unsigned regnum)`: if the `regnum` is in the `longopregs`, it is longop.