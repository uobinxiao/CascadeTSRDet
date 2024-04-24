import numpy as np
import logging


class Summarize:

    def __init__(self, stats, params, eval):
        self.stats = stats
        self.params = params
        self.eval = eval

        self.logger = logging.getLogger(__name__)

    def summarize(self, ap=1, iouThr=None, areaRng='all', maxDets=100):
        p = self.params
        iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
        titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
        typeStr = '(AP)' if ap == 1 else '(AR)'
        iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
            if iouThr is None else '{:0.2f}'.format(iouThr)

        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
        if ap == 1:
            # dimension of precision: [TxRxKxAxM]
            s = self.eval['precision']
            # IoU
            if iouThr is not None:
                t = np.where(np.isclose(iouThr, p.iouThrs))[0]  #Changed by Johan since == doesn't work well for float
                s = s[t]
            s = s[:, :, :, aind, mind]
        else:
            # dimension of recall: [TxKxAxM]
            s = self.eval['recall']
            if iouThr is not None:
                t = np.where(np.isclose(iouThr, p.iouThrs))[0]    #Changed by Johan since == doesn't work well for float
                s = s[t]
            s = s[:, :, aind, mind]
        if len(s[s > -1]) == 0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s > -1])
        # print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))

        # if self.logger:
        #     self.logger.info(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
        # print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))

        return mean_s

    def summarizeDets(self):
        # stats = np.zeros((16,))
        stats = [0] * 16    #Using list instead of np array for inference : Johan
        stats[0] = self.summarize(1)
        stats[1] = self.summarize(1, iouThr=.50, maxDets=self.params.maxDets[2])
        stats[2] = self.summarize(1, iouThr=.60, maxDets=self.params.maxDets[2])
        stats[3] = self.summarize(1, iouThr=.70, maxDets=self.params.maxDets[2])
        stats[4] = self.summarize(1, iouThr=.80, maxDets=self.params.maxDets[2])
        stats[5] = self.summarize(1, iouThr=.9, maxDets=self.params.maxDets[2])
        stats[6] = self.summarize(1, iouThr=.95, maxDets=self.params.maxDets[2])
        # stats[7] = self.summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
        # stats[8] = self.summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
        stats[7] = self.summarize(1, areaRng='large', maxDets=self.params.maxDets[2])

        stats[8] = self.summarize(0)  # MAR
        stats[9] = self.summarize(0, iouThr=.50, maxDets=self.params.maxDets[2])
        stats[10] = self.summarize(0, iouThr=.60, maxDets=self.params.maxDets[2])
        stats[11] = self.summarize(0, iouThr=.70, maxDets=self.params.maxDets[2])
        stats[12] = self.summarize(0, iouThr=.80, maxDets=self.params.maxDets[2])
        stats[13] = self.summarize(0, iouThr=.9, maxDets=self.params.maxDets[2])
        stats[14] = self.summarize(0, iouThr=.95, maxDets=self.params.maxDets[2])
        # stats[17] = self.summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
        # stats[18] = self.summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
        stats[15] = self.summarize(0, areaRng='large', maxDets=self.params.maxDets[2])
        return stats

    def summarizeDetsTNCR(self):
        stats = [0] * 24
        stats[0] = self.summarize(1)
        stats[1] = self.summarize(1, iouThr=.50, maxDets=self.params.maxDets[2])
        stats[2] = self.summarize(1, iouThr=.55, maxDets=self.params.maxDets[2])
        stats[3] = self.summarize(1, iouThr=.60, maxDets=self.params.maxDets[2])
        stats[4] = self.summarize(1, iouThr=.65, maxDets=self.params.maxDets[2])
        stats[5] = self.summarize(1, iouThr=.70, maxDets=self.params.maxDets[2])
        stats[6] = self.summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])
        stats[7] = self.summarize(1, iouThr=.80, maxDets=self.params.maxDets[2])
        stats[8] = self.summarize(1, iouThr=.85, maxDets=self.params.maxDets[2])
        stats[9] = self.summarize(1, iouThr=.9, maxDets=self.params.maxDets[2])
        stats[10] = self.summarize(1, iouThr=.95, maxDets=self.params.maxDets[2])
        # stats[7] = self.summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
        # stats[8] = self.summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
        stats[11] = self.summarize(1, areaRng='large', maxDets=self.params.maxDets[2])

        stats[12] = self.summarize(0)  # MAR
        stats[13] = self.summarize(0, iouThr=.50, maxDets=self.params.maxDets[2])
        stats[14] = self.summarize(0, iouThr=.55, maxDets=self.params.maxDets[2])
        stats[15] = self.summarize(0, iouThr=.60, maxDets=self.params.maxDets[2])
        stats[16] = self.summarize(0, iouThr=.65, maxDets=self.params.maxDets[2])
        stats[17] = self.summarize(0, iouThr=.70, maxDets=self.params.maxDets[2])
        stats[18] = self.summarize(0, iouThr=.75, maxDets=self.params.maxDets[2])
        stats[19] = self.summarize(0, iouThr=.80, maxDets=self.params.maxDets[2])
        stats[20] = self.summarize(0, iouThr=.85, maxDets=self.params.maxDets[2])
        stats[21] = self.summarize(0, iouThr=.9, maxDets=self.params.maxDets[2])
        stats[22] = self.summarize(0, iouThr=.95, maxDets=self.params.maxDets[2])
        # stats[17] = self.summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
        # stats[18] = self.summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
        stats[23] = self.summarize(0, areaRng='large', maxDets=self.params.maxDets[2])
        return stats
