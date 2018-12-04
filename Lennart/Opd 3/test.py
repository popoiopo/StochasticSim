try:
    from PySide import QtCore, QtGui
    HAVE_QT = True

except ImportError:
    print(
        "* * * * PySide module not available for display of chemical structure * * * * "
    )
    HAVE_QT = False

def chemParticleDynamics(bondDict,
                         numSteps=5000,
                         bondLen=1.0,
                         timeStep=0.01,
                         updateFunc=None):

    atoms = list(bondDict.keys())  # list() not needed in Python 2
    numAtoms = len(atoms)
    atomCoords = uniform(-10.0, 10.0, (numAtoms, 3))

    indices = range(numAtoms)
    n = float(numSteps)

    for step in range(numSteps):  # could use xrange in Python 2
        temp = exp(-step / n)

        if updateFunc:  # Extra for graphical display
            print("Step:", step)
            updateFunc(atomCoords)

        for i in indices[1:]:
            atom = atoms[i]
            coords = atomCoords[i]
            velocity = zeros(3, float)

            for j in indices:
                if i == j:
                    continue

                delta = coords - atomCoords[j]
                delta2 = delta * delta
                dist2 = delta2.sum()

                bound = bondDict[atoms[j]]
                if atom in bound:
                    force = bondLen - sqrt(dist2)

                else:
                    force = 1.0 / (dist2 * dist2)

                force = min(max(-200.0, force), 200.0)
                velocity += delta * force * temp * timeStep

            atomCoords[i] += velocity

    center = atomCoords.mean(axis=0)
    atomCoords = atomCoords - center

    return atomCoords

if HAVE_QT:

    class ChemView(QtGui.QWidget):
        def __init__(self, parent, topology):
            QtGui.QWidget.__init__(self, parent)

            self.topology = topology
            self.coords = None
            self.depthOfField = 5.0
            self.atomRadius = 30.0

            atoms = list(topology.keys())  # list() not needed in Python 2
            self.atoms = atoms
            iDict = dict([(x, i) for i, x in enumerate(atoms)])

            bondDict = {}
            for atom in atoms:
                bound = [iDict[a] for a in topology[atom]]
                bondDict[iDict[atom]] = bound

            self.bondDict = bondDict
            self.setMinimumHeight(600)
            self.setMinimumWidth(800)
            self.movePos = None

        def mousePressEvent(self, event):
            QtGui.QWidget.mousePressEvent(self, event)
            self.movePos = event.pos()

        def mouseMoveEvent(self, event):
            pos = event.pos()

            delta = self.movePos - pos
            dx = delta.x()
            dy = delta.y()

            rX = array(getRotationMatrix((0.0, 1.0, 0.0), dx * -0.01))
            rY = array(getRotationMatrix((1.0, 0.0, 0.0), dy * 0.01))

            c = self.coords
            c = dot(c, rX)
            c = dot(c, rY)

            self.movePos = pos
            self.updateCoords(c)

        def runDynamics(self):

            chemParticleDynamics(self.topology, updateFunc=self.updateCoords)

        def updateCoords(self, coords):

            self.coords = coords
            self.repaint()  # Calls paintEvent()

        def paintEvent(self, event):

            if self.coords is None:
                return

            scale = 50.0

            painter = QtGui.QPainter()
            painter.begin(self)

            dof = self.depthOfField
            rad = self.atomRadius

            cx = self.width() / 2.0
            cy = self.height() / 2.0

            p1 = QtGui.QColor(0, 0, 0)
            p2 = QtGui.QColor(64, 64, 64)

            setPen = painter.setPen

            painter.setBrush(QtGui.QColor(128, 128, 128, 128))

            drawEllipse = painter.drawEllipse
            drawText = painter.drawText
            setPen(p2)

            for i in self.bondDict.keys():
                x, y, z = self.coords[i]
                perpective = dof / (z - dof)
                xView = cx + scale * x * perpective
                yView = cy + scale * y * perpective

                for j in self.bondDict[i]:
                    x2, y2, z2 = self.coords[j]
                    perpective2 = dof / (z2 - dof)
                    xView2 = cx + scale * x2 * perpective2
                    yView2 = cy + scale * y2 * perpective2

                    painter.drawLine(xView, yView, xView2, yView2)

            sortCoords = [(z, i, x, y)
                          for i, (x, y, z) in enumerate(self.coords)]
            sortCoords.sort()

            QPointF = QtCore.QPointF
            for z, i, x, y in sortCoords:
                perpective = dof / (z - dof)
                xView = cx + scale * x * perpective
                yView = cy + scale * y * perpective
                rView = rad * perpective

                point = QPointF(xView, yView)
                setPen(p2)
                drawEllipse(point, rView, rView)
                setPen(p1)
                drawText(point, self.atoms[i])

            painter.end()