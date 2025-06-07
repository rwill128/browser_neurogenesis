// --- Spring Class (Soft Body) ---
class Spring {
    constructor(p1, p2, stiffness, dampingFactor, restLength = null, isRigid = false) {
        this.p1 = p1;
        this.p2 = p2;
        this.isRigid = isRigid;
        if (this.isRigid) {
            this.stiffness = RIGID_SPRING_STIFFNESS;
            this.dampingFactor = RIGID_SPRING_DAMPING;
        } else {
            this.stiffness = stiffness;
            this.dampingFactor = dampingFactor;
        }
        this.restLength = restLength === null ? p1.pos.sub(p2.pos).mag() : restLength;

        // Temporary vectors for calculations
        this._tempDiffPos = new Vec2();
        this._tempDirection = new Vec2();
        this._tempRelVel = new Vec2();
        this._tempP1Vel = new Vec2();
        this._tempP2Vel = new Vec2();
        this._tempSpringForceVec = new Vec2();
        this._tempDampingForceVec = new Vec2();
        this._tempTotalForceVec = new Vec2();
    }
    applyForce() {
        const diffPos = this._tempDiffPos.copyFrom(this.p1.pos).subInPlace(this.p2.pos);
        const currentLength = diffPos.mag(); // mag() still creates a new Vec2 for its internal calculation if not careful, but it returns a scalar. Let's assume Vec2.mag() is efficient or accept this one.
        if (currentLength === 0) return;
        const displacement = currentLength - this.restLength;
        const direction = this._tempDirection.copyFrom(diffPos).normalizeInPlace();

        const springForceMagnitude = -this.stiffness * displacement;
        const springForce = this._tempSpringForceVec.copyFrom(direction).mulInPlace(springForceMagnitude);

        const p1_vel_implicit = this._tempP1Vel.copyFrom(this.p1.pos).subInPlace(this.p1.prevPos);
        const p2_vel_implicit = this._tempP2Vel.copyFrom(this.p2.pos).subInPlace(this.p2.prevPos);
        const relVel_implicit = this._tempRelVel.copyFrom(p1_vel_implicit).subInPlace(p2_vel_implicit);

        const velAlongSpring = Vec2.dot(relVel_implicit, direction); // Vec2.dot is a static method, returns scalar
        const dampingForceMagnitude = -this.dampingFactor * velAlongSpring;
        const dampingForce = this._tempDampingForceVec.copyFrom(direction).mulInPlace(dampingForceMagnitude);

        const totalForce = this._tempTotalForceVec.copyFrom(springForce).addInPlace(dampingForce);
        this.p1.applyForce(totalForce);
        // For p2, we need to apply the negative of totalForce. 
        // We can reuse totalForce by scaling, applying, then scaling back if necessary, or use another temp if complex.
        // Or, simpler, just make sure applyForce(totalForce.mul(-1)) is efficient or this is acceptable.
        // Given applyForce now uses addInPlace, we should be fine with creating one new vector here for the negated force if MassPoint.applyForce takes a const ref or copies.
        // MassPoint.applyForce(f) { this.force.addInPlace(f); } means 'f' is not modified. So totalForce.mul(-1) which creates a new Vec2 is fine.
        // However, to be fully in-place for Spring's applyForce, we can do:
        // this.p2.applyForce(this._tempTotalForceVec.copyFrom(totalForce).mulInPlace(-1)); // This would modify _tempTotalForceVec for p2
        // But totalForce IS _tempTotalForceVec. So apply, then negate, then apply to p2.
        this.p2.applyForce(this._tempTotalForceVec.mulInPlace(-1)); // totalForce is already _tempTotalForceVec, negate it and apply.
    }
    draw(ctx) {
        ctx.lineCap = "round";
        ctx.lineJoin = "round";
        ctx.beginPath();
        ctx.moveTo(this.p1.pos.x, this.p1.pos.y);
        ctx.lineTo(this.p2.pos.x, this.p2.pos.y);
        ctx.strokeStyle = this.isRigid ? 'rgba(255, 255, 0, 0.9)' : 'rgba(150,150,150,0.6)'; // Bright yellow for rigid springs
        ctx.lineWidth = this.isRigid ? 3 : 2; // Thicker line for rigid springs
        ctx.stroke();
    }
} 