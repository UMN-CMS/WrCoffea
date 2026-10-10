c Fixed-order MG5_aMC analysis for the WR signal NLO/LO k-factor (AN-19-083 App. A method):
c the invariant mass of the heavy neutrino and the primary lepton, m(lN) = the WR* virtuality,
c filled for the full NLO result and for its Born (LO) part, plus the total rates.
c Histograms (HwU, written to Events/run_XX/MADatNLO.HwU):
c   1  total rate (NLO)        2  total rate (Born)
c   3  m(lN) (NLO)             4  m(lN) (Born)
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine analysis_begin(nwgt,weights_info)
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      implicit none
      integer nwgt
      character*(*) weights_info(*)
      call HwU_inithist(nwgt,weights_info)
      call HwU_book(1,'total rate      ', 5,0.5d0,5.5d0)
      call HwU_book(2,'total rate Born ', 5,0.5d0,5.5d0)
      call HwU_book(3,'mlN             ', 400,0d0,10000d0)
      call HwU_book(4,'mlN Born        ', 400,0d0,10000d0)
      return
      end


cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine analysis_end(dummy)
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      implicit none
      double precision dummy
      call HwU_write_file
      return
      end


cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine analysis_fill(p,istatus,ipdg,wgts,ibody)
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c Heavy neutrinos: 9900012/14/16; primary lepton: e or mu (+-11, +-13).
      implicit none
      include 'nexternal.inc'
      integer istatus(nexternal)
      integer iPDG(nexternal)
      double precision p(0:4,nexternal)
      double precision wgts(*)
      integer ibody
      integer i, in, il
      double precision e, px, py, pz, m2, mln
      in = 0
      il = 0
      do i = 1, nexternal
         if (istatus(i).eq.1) then
            if (abs(ipdg(i)).eq.9900012 .or. abs(ipdg(i)).eq.9900014
     &           .or. abs(ipdg(i)).eq.9900016) in = i
            if (abs(ipdg(i)).eq.11 .or. abs(ipdg(i)).eq.13) il = i
         endif
      enddo
      call HwU_fill(1,1d0,wgts)
      if (ibody.eq.3) call HwU_fill(2,1d0,wgts)
      if (in.eq.0 .or. il.eq.0) return
      e  = p(0,in) + p(0,il)
      px = p(1,in) + p(1,il)
      py = p(2,in) + p(2,il)
      pz = p(3,in) + p(3,il)
      m2 = e**2 - px**2 - py**2 - pz**2
      mln = sqrt(max(m2, 0d0))
      call HwU_fill(3,mln,wgts)
      if (ibody.eq.3) call HwU_fill(4,mln,wgts)
      return
      end
