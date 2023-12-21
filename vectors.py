from __future__ import annotations

class Vector(object):
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, other: Vector) -> Vector:
        return Vector(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: Vector) -> Vector:
        return Vector(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, other: float) -> Vector:
        return Vector(self.x * other, self.y * other, self.z * other)
    
    def __rmul__(self, other: float) -> Vector:
        return self.__mul__(other)
    
    def dot(self, other: Vector) -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z
   
    def squared(self) -> float:
        return self.dot(self)

    def to_list(self) -> list[float]:
        return [self.x, self.y, self.z]

    def __str__(self) -> str:
        return f"Vector({self.x:.16e}, {self.y:.16e}, {self.z:.16e})"
    
class LorentzVector(object):
    def __init__(self, t: float, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z
        self.t = t

    def __add__(self, other: LorentzVector) -> LorentzVector:
        return LorentzVector(self.t + other.t, self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: LorentzVector) -> LorentzVector:
        return LorentzVector(self.t - other.t, self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, other: float) -> LorentzVector:
        return LorentzVector(self.t * other, self.x * other, self.y * other, self.z * other)
    
    def __rmul__(self, other: float) -> LorentzVector:
        return self.__mul__(other)
    
    def dot(self, other: LorentzVector) -> float:
        return self.t * other.t -self.x * other.x -self.y * other.y -self.z * other.z
    
    def squared(self) -> float:
        return self.dot(self)
    
    def spatial_squared(self) -> float:
        return self.x**2 + self.y**2 + self.z**2

    def spatial(self) -> Vector:
        return Vector(self.x, self.y, self.z)

    def to_list(self) -> list[float]:
        return [self.t, self.x, self.y, self.z]

    def __str__(self) -> str:
        return f"LorentzVector({self.t:.16e}, {self.x:.16e}, {self.y:.16e}, {self.z:.16e})"