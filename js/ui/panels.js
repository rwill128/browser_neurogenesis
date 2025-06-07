// --- DOM Element Selections for Panels ---
const infoPanel = document.getElementById('infoPanel');
const closeInfoPanelButton = document.getElementById('closeInfoPanel');
const copyInfoPanelButton = document.getElementById('copyInfoPanelButton');
const allPointsInfoContainer = document.getElementById('allPointsInfoContainer');

const statsPanel = document.getElementById('statsPanel');
const toggleStatsPanelButton = document.getElementById('toggleStatsPanelButton');
const closeStatsPanelButton = document.getElementById('closeStatsPanelButton');
const copyStatsPanelButton = document.getElementById('copyStatsPanelButton');
const nodeTypeStatsDiv = document.getElementById('nodeTypeStats');


// --- Panel Update Functions ---

function updateInfoPanel() {
    if (selectedInspectBody && selectedInspectPoint) {
        document.getElementById('infoBodyId').textContent = selectedInspectBody.id;
        document.getElementById('infoBodyStiffness').textContent = selectedInspectBody.getAverageStiffness().toFixed(2);
        document.getElementById('infoBodyDamping').textContent = selectedInspectBody.getAverageDamping().toFixed(2);
        document.getElementById('infoBodyMotorInterval').textContent = selectedInspectBody.motorImpulseInterval;
        document.getElementById('infoBodyMotorCap').textContent = selectedInspectBody.motorImpulseMagnitudeCap.toFixed(2);
        document.getElementById('infoBodyEmitterStrength').textContent = selectedInspectBody.emitterStrength.toFixed(2);
        document.getElementById('infoBodyEmitterDirX').textContent = selectedInspectBody.emitterDirection.x.toFixed(2);
        document.getElementById('infoBodyEmitterDirY').textContent = selectedInspectBody.emitterDirection.y.toFixed(2);
        document.getElementById('infoBodyNumOffspring').textContent = selectedInspectBody.numOffspring;
        document.getElementById('infoBodyOffspringRadius').textContent = selectedInspectBody.offspringSpawnRadius.toFixed(1);
        document.getElementById('infoBodyPointAddChance').textContent = selectedInspectBody.pointAddChance.toFixed(3);
        document.getElementById('infoBodySpringConnectionRadius').textContent = selectedInspectBody.springConnectionRadius.toFixed(1);
        document.getElementById('infoBodyEnergy').textContent = selectedInspectBody.creatureEnergy.toFixed(2);
        document.getElementById('infoBodyReproEnergyThreshold').textContent = selectedInspectBody.reproductionEnergyThreshold;
        document.getElementById('infoBodyCurrentMaxEnergy').textContent = selectedInspectBody.currentMaxEnergy.toFixed(2);
        document.getElementById('infoBodyTicksBirth').textContent = selectedInspectBody.ticksSinceBirth;
        document.getElementById('infoBodyCanReproduce').textContent = selectedInspectBody.canReproduce;
        document.getElementById('infoBodyRewardStrategy').textContent = getRewardStrategyString(selectedInspectBody.rewardStrategy);
        document.getElementById('infoBodyEnergyPhoto').textContent = selectedInspectBody.energyGainedFromPhotosynthesis.toFixed(2);
        document.getElementById('infoBodyEnergyEat').textContent = selectedInspectBody.energyGainedFromEating.toFixed(2);
        document.getElementById('infoBodyEnergyPred').textContent = selectedInspectBody.energyGainedFromPredation.toFixed(2);

        document.getElementById('infoBodyCostBase').textContent = selectedInspectBody.energyCostFromBaseNodes.toFixed(2);
        document.getElementById('infoBodyCostEmitter').textContent = selectedInspectBody.energyCostFromEmitterNodes.toFixed(2);
        document.getElementById('infoBodyCostEater').textContent = selectedInspectBody.energyCostFromEaterNodes.toFixed(2);
        document.getElementById('infoBodyCostPredator').textContent = selectedInspectBody.energyCostFromPredatorNodes.toFixed(2);
        document.getElementById('infoBodyCostNeuron').textContent = selectedInspectBody.energyCostFromNeuronNodes.toFixed(2);
        document.getElementById('infoBodyCostSwimmer').textContent = selectedInspectBody.energyCostFromSwimmerNodes.toFixed(2);
        document.getElementById('infoBodyCostJet').textContent = selectedInspectBody.energyCostFromJetNodes.toFixed(2);
        document.getElementById('infoBodyCostAttractor').textContent = selectedInspectBody.energyCostFromAttractorNodes.toFixed(2);
        document.getElementById('infoBodyCostRepulsor').textContent = selectedInspectBody.energyCostFromRepulsorNodes.toFixed(2);
        document.getElementById('infoBodyCostPhoto').textContent = selectedInspectBody.energyCostFromPhotosyntheticNodes.toFixed(2);
        document.getElementById('infoBodyCostGrabbing').textContent = selectedInspectBody.energyCostFromGrabbingNodes.toFixed(2);
        document.getElementById('infoBodyCostEye').textContent = selectedInspectBody.energyCostFromEyeNodes.toFixed(2);
        
        let reproGeneEl = document.getElementById('infoBodyReproCooldownGeneVal');
        if (reproGeneEl) reproGeneEl.textContent = selectedInspectBody.reproductionCooldownGene;

        let effectiveReproEl = document.getElementById('infoBodyEffectiveReproCooldownVal');
        if (effectiveReproEl) effectiveReproEl.textContent = selectedInspectBody.effectiveReproductionCooldown;

        allPointsInfoContainer.innerHTML = '<h5>All Mass Points</h5>';
        selectedInspectBody.massPoints.forEach((point, index) => {
            const pointEntryDiv = document.createElement('div');
            pointEntryDiv.className = 'point-info-entry';
            let content = `<p><strong>Point Index:</strong> ${index}</p><p><strong>Node Type:</strong> ${getNodeTypeString(point.nodeType)}</p><p><strong>Movement Type:</strong> ${getMovementTypeString(point.movementType)}</p><p><strong>Mass:</strong> ${point.mass.toFixed(2)}</p><p><strong>Radius:</strong> ${point.radius.toFixed(2)}</p><p><strong>World Pos:</strong> X: ${point.pos.x.toFixed(2)}, Y: ${point.pos.y.toFixed(2)}</p><p><strong>Can Be Grabber:</strong> ${point.canBeGrabber}</p>`;
            if (point.nodeType === NodeType.EMITTER) content += `<p><strong>Dye Color:</strong> R:${point.dyeColor[0].toFixed(0)} G:${point.dyeColor[1].toFixed(0)} B:${point.dyeColor[2].toFixed(0)}</p>`;
            if (point.nodeType === NodeType.JET) content += `<p><strong>Max Effective Velocity:</strong> ${point.maxEffectiveJetVelocity.toFixed(2)}</p>`;
            if (point.isGrabbing) content += `<p><strong>State:</strong> Grabbing</p>`; else content += `<p><strong>State:</strong> Normal</p>`;
            if (point.nodeType === NodeType.EYE) {
                content += `<h6>Eye Sensor Data:</h6><p><strong>Target Type:</strong> ${getEyeTargetTypeString(point.eyeTargetType)}</p><p><strong>Sees Target:</strong> ${point.seesTarget}</p>`;
                if (point.seesTarget) content += `<p><strong>Target Distance:</strong> ${(point.nearestTargetMagnitude * EYE_DETECTION_RADIUS).toFixed(1)} (norm: ${point.nearestTargetMagnitude.toFixed(3)})</p><p><strong>Target Angle:</strong> ${(point.nearestTargetDirection * 180 / Math.PI).toFixed(1)}&deg;</p>`;
            }
            if (point.nodeType === NodeType.NEURON && point.neuronData) {
                if (point.neuronData.isBrain) {
                    content += `<h6>Brain Details:</h6><p><strong>Role:</strong> Active Brain</p><p><strong>Hidden Layer Size:</strong> ${point.neuronData.hiddenLayerSize || 'N/A'}</p><p><strong>Input Vector Size:</strong> ${point.neuronData.inputVectorSize || 'N/A'}</p><p><strong>Output Vector Size:</strong> ${point.neuronData.outputVectorSize || 'N/A'}</p>`;
                    if (typeof point.neuronData.lastAvgNormalizedReward === 'number') content += `<p><strong>Avg Batch Reward:</strong> ${point.neuronData.lastAvgNormalizedReward.toFixed(3)}</p>`;
                    if (point.neuronData.currentFrameInputVectorWithLabels && point.neuronData.currentFrameInputVectorWithLabels.length > 0) {
                        content += `<h6>Brain Inputs (Real-time):</h6>`;
                        point.neuronData.currentFrameInputVectorWithLabels.forEach(input => { content += `<p><strong style="color:#aadeff;">${input.label}:</strong> <span class="stat-value">${input.value.toFixed(3)}</span></p>`; });
                    }
                    if (point.neuronData.currentFrameActionDetails && point.neuronData.currentFrameActionDetails.length > 0) {
                        content += `<h6>Brain Actions (Real-time Outputs):</h6>`;
                        point.neuronData.currentFrameActionDetails.forEach(action => {
                            let finalValueDisplay = sigmoid(action.sampledAction).toFixed(3);
                            if (action.label.includes("Direction")) finalValueDisplay = `${(action.sampledAction).toFixed(2)} rad`;
                            else if (action.label.includes("Toggle")) finalValueDisplay = sigmoid(action.sampledAction) > 0.5 ? "ON" : "OFF";
                            content += `<p><strong style="color:#aadeff;">${action.label}:</strong> <span class="stat-value">${finalValueDisplay}</span> <em style="font-size:0.9em; color:#999;">(&mu;:${action.mean.toFixed(2)}, &sigma;:${action.stdDev.toFixed(2)})</em></p>`;
                        });
                    }
                } else {
                    content += `<h6>Neuron (Non-Brain)</h6><p><strong>Hidden Layer Size (if applicable):</strong> ${point.neuronData.hiddenLayerSize || 'N/A'}</p>`;
                }
            }
            pointEntryDiv.innerHTML = content;
            allPointsInfoContainer.appendChild(pointEntryDiv);
        });

        if (!infoPanel.classList.contains('open')) {
            infoPanel.classList.add('open');
        }
    } else {
        allPointsInfoContainer.innerHTML = '';
        // Reset all info panel fields to '-'
        const infoSpans = infoPanel.querySelectorAll('.info-section span[id]');
        infoSpans.forEach(span => span.textContent = '-');
        infoPanel.classList.remove('open');
    }
}

function updateStatsPanel() {
    if (!nodeTypeStatsDiv) return;
    const mutationTypeStatsDiv = document.getElementById('mutationTypeStats');
    const globalEnergyGainsStatsDiv = document.getElementById('globalEnergyGainsStats');
    const globalEnergyCostsStatsDiv = document.getElementById('globalEnergyCostsStats');

    // Node Type Proportions
    nodeTypeStatsDiv.innerHTML = '<p><strong>Node Type Proportions:</strong></p>';
    const nodeCounts = {};
    let totalNodes = 0;
    for (const typeName in NodeType) { nodeCounts[NodeType[typeName]] = 0; }
    softBodyPopulation.forEach(body => {
        body.massPoints.forEach(point => {
            if (nodeCounts[point.nodeType] !== undefined) nodeCounts[point.nodeType]++;
            totalNodes++;
        });
    });
    
    if (totalNodes === 0) {
        nodeTypeStatsDiv.innerHTML += "<p>No creatures to analyze.</p>";
    } else {
        for (const typeName in NodeType) {
            const typeEnum = NodeType[typeName];
            const count = nodeCounts[typeEnum];
            const percentage = ((count / totalNodes) * 100).toFixed(2);
            const typeString = getNodeTypeString(typeEnum);
            const statLineContainer = document.createElement('div');
            statLineContainer.className = 'stat-line-container';
            statLineContainer.innerHTML = `<p style="margin: 0;"><strong>${typeString}:</strong> <span class="stat-value">${count} (${percentage}%)</span></p>`;
            if (count > 0) {
                const button = document.createElement('button');
                button.textContent = '🔍';
                button.className = 'stats-panel-button';
                button.title = `Find next creature with a ${typeString} node`;
                button.onclick = () => handleNodeTypeLabelClick(typeName);
                statLineContainer.appendChild(button);
            }
            nodeTypeStatsDiv.appendChild(statLineContainer);
        }
        nodeTypeStatsDiv.innerHTML += `<p style="margin-top: 8px;"><strong>Total Nodes:</strong> <span class="stat-value">${totalNodes}</span></p>`;
    }

    // Mutation Type Counts
    if (mutationTypeStatsDiv) {
        let mutationStatsHTML = "<p><strong>Mutation Occurrences:</strong></p>";
        let totalMutations = Object.values(mutationStats).reduce((sum, count) => sum + count, 0);
        for (const key in mutationStats) {
            mutationStatsHTML += `<p><strong>${key.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase())}:</strong> <span class="stat-value">${mutationStats[key]}</span></p>`;
        }
        mutationStatsHTML += `<p><strong>Total Mutations Tracked:</strong> <span class="stat-value">${totalMutations}</span></p>`;
        mutationTypeStatsDiv.innerHTML = mutationStatsHTML;
    }

    // Global Energy Gains/Costs
    if (globalEnergyGainsStatsDiv) globalEnergyGainsStatsDiv.innerHTML = `<p><strong>Global Energy Gains (All Time):</strong></p><p><strong>Photosynthesis:</strong> <span class="stat-value">${globalEnergyGains.photosynthesis.toFixed(2)}</span></p><p><strong>Eating:</strong> <span class="stat-value">${globalEnergyGains.eating.toFixed(2)}</span></p><p><strong>Predation:</strong> <span class="stat-value">${globalEnergyGains.predation.toFixed(2)}</span></p>`;
    if (globalEnergyCostsStatsDiv) {
        let costsHTML = "<p><strong>Global Energy Costs (All Time):</strong></p>";
        for(const key in globalEnergyCosts) {
            costsHTML += `<p><strong>${key.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase())}:</strong> <span class="stat-value">${globalEnergyCosts[key].toFixed(2)}</span></p>`;
        }
        globalEnergyCostsStatsDiv.innerHTML = costsHTML;
    }
}


function initializePanelListeners() {
    toggleStatsPanelButton.onclick = function () {
        statsPanel.classList.toggle('open');
        if (statsPanel.classList.contains('open')) {
            updateStatsPanel();
        }
    }
    
    closeStatsPanelButton.onclick = () => statsPanel.classList.remove('open');
    
    copyStatsPanelButton.onclick = function () {
        const textToCopy = [
            document.getElementById('nodeTypeStats').innerText,
            document.getElementById('mutationTypeStats').innerText,
            document.getElementById('globalEnergyGainsStats').innerText,
            document.getElementById('globalEnergyCostsStats').innerText
        ].join('\n\n');
        
        navigator.clipboard.writeText(textToCopy.trim()).then(() => {
            const originalText = copyStatsPanelButton.textContent;
            copyStatsPanelButton.textContent = "Copied!";
            setTimeout(() => { copyStatsPanelButton.textContent = originalText; }, 1500);
        }).catch(err => {
            console.error('Failed to copy stats text: ', err);
        });
    }

    closeInfoPanelButton.onclick = () => {
        infoPanel.classList.remove('open');
        selectedInspectBody = null;
        selectedInspectPoint = null;
    }

    copyInfoPanelButton.onclick = function () {
        if (!selectedInspectBody) return;
        let infoText = "";
        const panelElements = infoPanel.querySelectorAll('.info-section p, .info-section h5, .point-info-entry p, .point-info-entry h6');
        panelElements.forEach(el => {
            if (el.tagName === 'H5' || el.tagName === 'H6') {
                infoText += el.textContent.trim() + "\n";
            } else if (el.tagName === 'P') {
                const strongTag = el.querySelector('strong');
                if (strongTag) {
                    let label = strongTag.textContent.trim();
                    let value = el.textContent.substring(label.length).trim().replace(/^:/, '').trim();
                    infoText += label + ": " + value + "\n";
                } else {
                    infoText += el.textContent.trim() + "\n";
                }
            }
        });
        navigator.clipboard.writeText(infoText.trim()).then(() => {
            copyInfoPanelButton.textContent = "Copied!";
            setTimeout(() => { copyInfoPanelButton.textContent = "Copy"; }, 1500);
        }).catch(err => console.error('Failed to copy text: ', err));
    }
} 