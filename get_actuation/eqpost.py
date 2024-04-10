#!/APP/anaconda3/bin/python3
import os, sys
import time
import numpy as np
import matplotlib.pyplot as plt
from shutil import move, copyfile, copytree
from scipy.interpolate import interp1d, interp2d
from skimage import measure
import eqpost as eqp


class ineqdsk:

    if __name__ == "__main__":
        ineqdsk = eqp.ineqdsk()

    def __init__(self):

        self.do_plot = True
        self.eqdsk_file = "d3d_efitrt2_196114_003009.geqdsk"

        self._make_equil_infos()

    def _make_equil_infos(self):

        self._preset_device()
        self._read_gfile()
        self._get_xpoint()
        self._get_striking_point()

        print(
            ">>> RZ-Xpoint upper [%f,%f]/ lower [%f,%f] in [m]"
            % (self.xpnt_R[0], self.xpnt_Z[0], self.xpnt_R[1], self.xpnt_Z[1])
        )
        if self.xpnt_P[0] > self.xpnt_P[1]:
            print(">>> Active xpoint = lower")
        if self.xpnt_P[0] < self.xpnt_P[1]:
            print(">>> Active xpoint = upper")
        print(
            ">>> STK-point inner [%f,%f]/ outer [%f,%f] in [m]"
            % (self.stk_in[0], self.stk_in[1], self.stk_out[0], self.stk_out[1])
        )

        if self.do_plot:
            self._draw_equils()

    def _preset_device(self):

        # CCW is positive
        self.ipsign = +1.0
        # CCW
        self.btsign = +1.0
        # CCW
        self.vtsign = +1.0
        # CCW

        self.div_low_R = [
            +1.017,
            +1.017,
            +1.165,
            +1.368,
            +1.368,
            +1.744,
            +1.785,
            +2.134,
        ]
        self.div_low_Z = [
            +0.000,
            -1.218,
            -1.360,
            -1.360,
            -1.245,
            -1.245,
            -1.160,
            -0.947,
        ]

    def _read_gfile(self):
        print(">>> Read g-file")
        self.eq = eqp.eqdsk(self.eqdsk_file)

        # Sign corr
        self.eq.ip = self.eq.ip * self.ipsign
        self.eq.smag = self.eq.smag * self.ipsign
        self.eq.sbdy = self.eq.sbdy * self.ipsign
        self.eq.psirz = self.eq.psirz * self.ipsign
        self.eq.ffp = self.eq.ffp * self.ipsign

    def _get_xpoint(self):

        dr = 1.0e-4
        dz = 1.0e-4
        eps = 1.0e-4
        self.xpnt_R = np.zeros(2)
        self.xpnt_Z = np.zeros(2)
        self.xpnt_P = np.zeros(2)

        zmax = np.max(self.eq.rzbdy[:, 1])
        zmaxloc = np.argmax(self.eq.rzbdy[:, 1])
        rmax = self.eq.rzbdy[zmaxloc, 0]
        zmin = np.min(self.eq.rzbdy[:, 1])
        zminloc = np.argmin(self.eq.rzbdy[:, 1])
        rmin = self.eq.rzbdy[zminloc, 0]

        bnf = interp2d(self.eq.r, self.eq.z, self.eq.br**2 + self.eq.bz**2)

        # find upper-xpnt
        epx = 1.0e0
        epz = 1.0e0
        xx = rmax
        zz = zmax
        count1 = 0
        count2 = 0
        while (epx > eps or epz > eps) and count1 < 100:
            zz2 = np.linspace(
                zmax, 0.95 * self.eq.z[-1], int((0.95 * self.eq.z[-1] - zmax) / dz)
            )
            bnt = bnf(xx, zz2)
            minb = np.min(bnt)
            minbloc = np.argmin(bnt)
            epz = (zz2[minbloc] - zz) / zz
            zz = zz2[minbloc]

            xx2 = np.linspace(
                1.05 * self.eq.r[0],
                0.95 * self.eq.r[-1],
                int((0.95 * self.eq.r[-1] - 1.05 * self.eq.r[0]) / dr),
            )
            bnt = bnf(xx2, zz)
            minb = np.min(bnt)
            minbloc = np.argmin(bnt)
            epx = (xx2[minbloc] - xx) / xx
            xx = xx2[minbloc]
            count1 += 1

        self.xpnt_R[0] = xx
        self.xpnt_Z[0] = zz
        self.xpnt_P[0] = self.eq.psif(self.xpnt_R[0], self.xpnt_Z[0])

        # find lower-xpnt
        epx = 1.0e0
        epz = 1.0e0
        xx = rmin
        zz = zmin
        while (epx > eps or epz > eps) and count2 < 100:

            zz2 = np.linspace(
                1.05 * self.eq.z[0], zmin, int((zmin - 1.05 * self.eq.z[0]) / dz)
            )
            bnt = bnf(xx, zz2)
            minb = np.min(bnt)
            minbloc = np.argmin(bnt)
            epz = (zz2[minbloc] - zz) / zz
            zz = zz2[minbloc]

            xx2 = np.linspace(
                1.05 * self.eq.r[0],
                0.95 * self.eq.r[-1],
                int((0.95 * self.eq.r[-1] - 1.05 * self.eq.r[0]) / dr),
            )
            bnt = bnf(xx2, zz)
            minb = np.min(bnt)
            minbloc = np.argmin(bnt)
            epx = (xx2[minbloc] - xx) / xx
            xx = xx2[minbloc]
            count2 += 1

        if count1 == 100:
            print(">>> Failed to find upper xpoint")
        if count2 == 100:
            print(">>> Failed to find lower xpoint")
        self.xpnt_R[1] = xx
        self.xpnt_Z[1] = zz
        self.xpnt_P[1] = self.eq.psif(self.xpnt_R[1], self.xpnt_Z[1])
        self.xpnt_P = (self.xpnt_P - self.eq.smag) / (self.eq.sbdy - self.eq.smag)

    def _get_striking_point(self):

        lower_singlenull = False

        if self.xpnt_P[0] > self.xpnt_P[1]:
            lower_singlenull = True
            self.active_xpnt = [self.xpnt_R[1], self.xpnt_Z[1]]
        else:
            self.active_xpnt = [self.xpnt_R[0], self.xpnt_Z[0]]

        psi_bnd1 = min(self.xpnt_P[0], self.xpnt_P[1]) + 0.001
        psi_bnd2 = max(self.xpnt_P[0], self.xpnt_P[1]) + 0.001

        cs = self.eq._contour(psi_bnd1)
        tarv = 0.0
        ttarc = np.zeros((1, 2))
        for cc in cs:
            if lower_singlenull:
                if np.min(cc[:, 0]) < tarv:
                    self.lcfs = cc
                    tarv = np.min(cc[:, 0])
            else:
                if np.max(cc[:, 0]) > tarv:
                    self.lcfs = cc
                    tarv = np.max(cc[:, 0])

        self.stk_out = [0.0, 0.0, 100]
        self.stk_in = [0.0, 0.0, 100]
        for idiv in range(len(self.div_low_R) - 1):

            rr = np.linspace(self.div_low_R[idiv], self.div_low_R[idiv + 1], 101)
            zz = np.linspace(self.div_low_Z[idiv], self.div_low_Z[idiv + 1], 101)

            for ielm in range(101):

                ll = (rr[ielm] - self.lcfs[:, 1]) ** 2 + (
                    zz[ielm] - self.lcfs[:, 0]
                ) ** 2

                imin = np.argmin(ll)

                if rr[ielm] > self.active_xpnt[0]:
                    if ll[imin] < self.stk_out[2]:
                        self.stk_out = [rr[ielm], zz[ielm], ll[imin]]
                else:
                    if ll[imin] < self.stk_in[2]:
                        self.stk_in = [rr[ielm], zz[ielm], ll[imin]]

        return

    def _draw_equils(self):

        fig = plt.figure("EFIT overview", figsize=(5, 8))
        ax1 = plt.subplot2grid((1, 1), (0, 0), rowspan=2)

        ax1.contourf(self.eq.R, self.eq.Z, self.eq.psirz, 50)
        ax1.plot(self.eq.rzbdy[:, 0], self.eq.rzbdy[:, 1], c="r")
        ax1.plot(self.eq.rzlim[:, 0], self.eq.rzlim[:, 1], c="blue")

        lower_singlenull = False
        if self.xpnt_P[0] > self.xpnt_P[1]:
            lower_singlenull = True
        psi_bnd1 = min(self.xpnt_P[0], self.xpnt_P[1]) + 0.001
        psi_bnd2 = max(self.xpnt_P[0], self.xpnt_P[1]) + 0.001

        # LCFS
        #        for cc in cs:
        #            if lower_singlenull: ind = np.where(cc[:,0] < self.xpnt_Z[0])
        #            else: ind = np.where(cc[:,0] > self.xpnt_Z[1])
        #            ax1.plot(cc[:,1][ind],cc[:,0][ind],c='red')

        #        cs = self.eq._contour(psi_bnd2);
        #        for cc in cs: ax1.plot(cc[:,1],cc[:,0],c='red')

        ax1.plot(self.lcfs[:, 1], self.lcfs[:, 0], c="yellow")

        ax1.scatter(self.eq.rmag, self.eq.zmag, marker="x", s=50, color="green")
        ax1.scatter(self.xpnt_R[0], self.xpnt_Z[0], marker="x", s=50, color="orange")
        ax1.scatter(self.xpnt_R[1], self.xpnt_Z[1], marker="x", s=50, color="orange")
        ax1.scatter(self.stk_out[0], self.stk_out[1], marker="x", s=50, color="red")
        ax1.scatter(self.stk_in[0], self.stk_in[1], marker="x", s=50, color="red")

        ax1.set_xlabel("R [m]")
        ax1.set_ylabel("Z [m]")
        fig.tight_layout()
        plt.show(block=True)


class eqdsk:
    def __init__(self, filename):
        self.filename = filename
        self._read_eqdsk(self.filename)
        self._make_grid()
        self._make_rho_R_psin()
        self._construct_volume()
        print(">>> Construct Poloidal fields")
        self._construct_2d_field()

    def _read_1d(self, file, num):

        dat = np.zeros(num)

        linen = int(np.floor(num / 5))

        ii = 0

        for i in range(linen):

            line = file.readline()

            for j in range(5):

                dat[ii] = float(line[16 * j : 16 * (j + 1)])
                ii = ii + 1

        if not (num == 5 * linen):
            line = file.readline()

        for i in range(num - 5 * linen):

            dat[ii] = float(line[16 * i : 16 * (i + 1)])
            ii = ii + 1

        return dat

    def _read_colum(self, file, num):

        dat = np.zeros(shape=(num, 2))

        dat1 = self._read_1d(file, num * 2)

        for i in range(num):

            dat[i, 0] = dat1[2 * i]
            dat[i, 1] = dat1[2 * i + 1]

        return dat

    def _read_2d(self, file, num1, num2):

        dat = np.zeros(shape=(num1, num2))

        dat1 = self._read_1d(file, num1 * num2)

        ii = 0

        for i in range(num1):

            for j in range(num2):

                dat[i, j] = dat1[ii]
                ii = ii + 1

        return dat

    def _read_eqdsk(self, filename):

        file = open(filename, "r")

        line = file.readline().split()  #
        linen = len(line)

        self.id = []

        if linen > 4:
            for i in range(linen - 4):
                self.id.append(line[i])
        self.shotn = 0
        try:
            if self.id[0].find("EFIT") > -1:
                self.shotn = int(self.id[3])
        except:
            pass

        self.idum = int(line[linen - 3])
        self.nw = int(line[linen - 2])
        self.nh = int(line[linen - 1])

        line = file.readline()  #
        self.rdim = float(line[0:16])
        self.zdim = float(line[16:32])
        self.rcentr = float(line[32:48])
        self.rleft = float(line[48:64])
        self.zmid = float(line[64:80])

        line = file.readline()  #
        self.rmag = float(line[0:16])
        self.zmag = float(line[16:32])
        self.smag = float(line[32:48])
        self.sbdy = float(line[48:64])
        self.bcentr = float(line[64:80])

        line = file.readline()  #
        self.ip = float(line[0:16])
        self.xdum = float(line[32:48])

        line = file.readline()  #

        self.fpol = self._read_1d(file, self.nw)
        self.pres = self._read_1d(file, self.nw)
        self.ffp = self._read_1d(file, self.nw)
        self.pp = self._read_1d(file, self.nw)
        self.psirz = self._read_2d(file, self.nh, self.nw)
        self.q = self._read_1d(file, self.nw)

        line = file.readline().split()
        self.nbbbs = int(line[0])
        self.limitr = int(line[1])

        self.rzbdy = self._read_colum(file, self.nbbbs)
        self.rzlim = self._read_colum(file, self.limitr)

        self.rzbdy[:, 1] = self.rzbdy[:, 1]
        self.rzlim[:, 1] = self.rzlim[:, 1]

        return

    def _make_grid(self):

        self.R = np.zeros(self.nw)
        self.Z = np.zeros(self.nh)
        self.psin = np.linspace(0, 1, self.nw)

        for i in range(self.nw):
            self.R[i] = self.rleft + float(i) * self.rdim / (self.nw - 1)
        for i in range(self.nh):
            self.Z[i] = (
                self.zmid + float(i) * self.zdim / (self.nh - 1) - self.zdim / 2.0
            )
            self.Z[i] = self.Z[i]

        self.RR, self.ZZ = np.meshgrid(self.R, self.Z)

        return

    def _contour(self, psi_norm):
        psi = psi_norm * (self.sbdy - self.smag) + self.smag
        cs = measure.find_contours(self.psirz, psi)
        for cc in cs:
            cc[:, 1] = cc[:, 1] * (max(self.R) - min(self.R)) / (
                len(self.R) - 1.0
            ) + min(self.R)
            cc[:, 0] = cc[:, 0] * (max(self.Z) - min(self.Z)) / (
                len(self.Z) - 1.0
            ) + min(self.Z)
        rz = np.copy(cs[0])
        rz[:, 0] = cs[0][:, 1]
        rz[:, 1] = cs[0][:, 0]

        return cs

    def _make_rho_R_psin(self):

        self.prhoR = np.zeros(shape=(201, 4))

        self.prhoR[:, 0] = np.linspace(0, 1.0, 201)

        RR = np.linspace(self.rmag, max(self.R) * 0.999, 301)
        RR2 = np.linspace(min(self.R) * 1.001, self.rmag, 301)

        psif = interp2d(self.R, self.Z, self.psirz)

        psir = psif(RR, self.zmag)
        psir2 = psif(RR2, self.zmag)

        psirn = np.zeros(301)
        psirn2 = np.zeros(301)

        for i in range(301):
            psirn[i] = (psir[i] - self.smag) / (self.sbdy - self.smag)
            psirn2[i] = (psir2[i] - self.smag) / (self.sbdy - self.smag)

        psirn[0] = 0.0
        psirn2[-1] = 0.0

        prf = interp1d(psirn, RR, "cubic")
        prf2 = interp1d(psirn2, RR2, "cubic")

        self.prhoR[:, 2] = prf(self.prhoR[:, 0])
        self.prhoR[0, 2] = self.rmag

        self.prhoR[:, 3] = prf2(self.prhoR[:, 0])
        self.prhoR[0, 3] = self.rmag

        qf = interp1d(self.psin, self.q, "cubic")
        q = qf(self.prhoR[:, 0])

        for i in range(200):
            self.prhoR[i + 1, 1] = np.trapz(q[0 : i + 2], x=self.prhoR[0 : i + 2, 0])
        for i in range(201):
            self.prhoR[i, 1] = np.sqrt(self.prhoR[i, 1] / self.prhoR[-1, 1])

        rhof = interp1d(self.prhoR[:, 0], self.prhoR[:, 1], "slinear")
        self.rho = rhof(self.psin)

        return

    def _construct_volume(self):

        len1 = len(self.rzbdy)
        len2 = 101
        psin = np.linspace(0, 1.0, len2)
        self.avolp = np.zeros(shape=(len2, 3))

        self.psif = interp2d(self.R, self.Z, self.psirz, "cubic")
        r = np.zeros(shape=(len1, len2))
        z = np.zeros(shape=(len1, len2))
        for i in range(len1):
            rr = np.linspace(self.rmag, self.rzbdy[i, 0], len2)
            zz = (self.rzbdy[i, 1] - self.zmag) / (self.rzbdy[i, 0] - self.rmag) * (
                rr - self.rmag
            ) + self.zmag
            psi = np.zeros(len2)
            for j in range(len2):
                psi[j] = (self.psif(rr[j], zz[j]) - self.smag) / (self.sbdy - self.smag)

            psi[0] = 0.0
            psi[-1] = 1.0
            psifr = interp1d(psi, rr, "cubic")
            r[i, :] = psifr(psin)
            z[i, :] = (self.rzbdy[i, 1] - self.zmag) / (
                self.rzbdy[i, 0] - self.rmag
            ) * (r[i, :] - self.rmag) + self.zmag

        sum1 = 0.0
        sum2 = 0.0
        sum3 = 0.0
        for i in range(len2 - 1):
            for j in range(len1):
                i1 = i + 1
                j1 = j + 1
                if j1 == len1:
                    j1 = 0

                dx1 = r[j1, i1] - r[j, i]
                dz1 = z[j1, i1] - z[j, i]
                dx2 = r[j1, i] - r[j, i1]
                dz2 = z[j1, i] - z[j, i1]
                dx3 = r[j1, i1] - r[j, i1]
                dz3 = r[j1, i1] - r[j, i1]

                dl1 = np.sqrt(dx1**2 + dz1**2)
                dl2 = np.sqrt(dx2**2 + dz2**2)
                cos = (dx1 * dx2 + dz1 * dz2) / dl1 / dl2

                if abs(cos) > 1:
                    cos = 1.0
                sin = np.sqrt(1.0 - cos**2)
                dA = 0.5 * dl1 * dl2 * sin
                Rc = 0.25 * (r[j, i] + r[j1, i] + r[j, i1] + r[j1, i1])
                Zc = 0.25 * (z[j, i] + z[j1, i] + z[j, i1] + z[j1, i1])
                sum1 = sum1 + dA
                sum2 = sum2 + 2.0 * np.pi * Rc * dA

            self.avolp[i + 1, 0] = psin[i + 1]
            self.avolp[i + 1, 1] = sum1
            self.avolp[i + 1, 2] = sum2

        pref = interp1d(self.psin, self.pres, "cubic")
        pres = pref(psin)

        self.wmhd = np.trapz(pres, x=self.avolp[:, 2]) * 1.5
        self.area = self.avolp[-1, 1]
        self.vol = self.avolp[-1, 2]
        self.pva = self.wmhd / 1.5 / self.avolp[-1, 2]

        return

    def _construct_2d_field(self):

        self.br = np.zeros(shape=(self.nh - 2, self.nw - 2))
        self.bz = np.copy(self.br)
        self.bt = np.copy(self.br)
        self.r = np.zeros(self.nw - 2)
        self.z = np.zeros(self.nh - 2)

        for i in range(self.nh - 2):
            self.z[i] = self.Z[i + 1]
        for i in range(self.nw - 2):
            self.r[i] = self.R[i + 1]

        fpf = interp1d(self.psin, self.fpol, "cubic")

        for i in range(self.nh - 2):
            Z = self.Z[i + 1]
            for j in range(self.nw - 2):
                R = self.R[j + 1]
                psi = (self.psirz[i + 1, j + 1] - self.smag) / (self.sbdy - self.smag)
                if psi < 0.0:
                    psi = 0.0

                self.br[i, j] = (
                    -(self.psirz[i + 2, j + 1] - self.psirz[i, j + 1])
                    / (self.Z[i + 2] - self.Z[i])
                    / R
                )
                self.bz[i, j] = (
                    +(self.psirz[i + 1, j + 2] - self.psirz[i + 1, j])
                    / (self.R[j + 2] - self.R[j])
                    / R
                )

                if psi < 1.0:
                    self.bt[i, j] = fpf(psi) / R
                else:
                    self.bt[i, j] = self.fpol[-1] / R
        return
