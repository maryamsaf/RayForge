//"C:\Users\GhaznaAli\Semester 8\SoftwareEngineering\PROJECT\rayforge-frontend\codes\script.js"

// 3D Scene Setup for embedded canvas
const canvas = document.getElementById('lung-canvas');

if (canvas) {
    const parentContainer = canvas.parentElement;
    
    // We want the canvas to be the size of the container, but responsive.
    const scene = new THREE.Scene();
    
    // Camera
    const camera = new THREE.PerspectiveCamera(40, parentContainer.clientWidth / parentContainer.clientHeight, 0.1, 1000);
    camera.position.z = 75; // Set back to a normal framing distance since container is now huge
    camera.position.y = 0;
    
    // Renderer
    const renderer = new THREE.WebGLRenderer({
        canvas: canvas,
        alpha: true, // Transparent background to match any theme
        antialias: true
    });
    renderer.setSize(parentContainer.clientWidth, parentContainer.clientHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    
    // --- Procedural 3D Lung Particle System ---
    const particleCount = 80000;
    const positions = new Float32Array(particleCount * 3);
    const colors = new Float32Array(particleCount * 3);
    const sizes = new Float32Array(particleCount);
    
    // Theme colors: Refactr Medic Reds and whites
    const colorRed = new THREE.Color(0xE8251A);
    const colorDarkRed = new THREE.Color(0xC01A10);
    const colorWhite = new THREE.Color(0xffffff);
    const colorBlack = new THREE.Color(0x0e0e0e);
    
    for(let i = 0; i < particleCount; i++) {
        const randType = Math.random();
        let x, y, z;
        
        if (randType > 0.85) {
            // Trachea / Bronchi network
            const isBronchi = Math.random() > 0.35;
            if (!isBronchi) {
                // Trachea (main central windpipe)
                y = Math.random() * 18 + 17;
                const rTub = Math.random() * 2.8;
                const th = Math.random() * Math.PI * 2;
                x = rTub * Math.cos(th);
                z = rTub * Math.sin(th);
            } else {
                // Main bronchi branching out diagonally into the lobes
                const isLeft = Math.random() > 0.5;
                const xSign = isLeft ? -1 : 1;
                const t = Math.random(); // 0 to 1
                // Add a little curve to the bronchi
                x = xSign * (t * 11 + Math.sin(t*Math.PI)*2);
                y = 18 - t * 9 + (Math.random()-0.5) * 2;
                z = (Math.random() - 0.5) * 3;
            }
        } else {
            // Main lung lobes geometry
            const isLeft = Math.random() < 0.5;
            const xSign = isLeft ? -1 : 1;
            
            // Focus volume mostly on outer shells with some inner depth
            const u = Math.pow(Math.random(), 0.85); 
            const v = Math.random();
            const theta = u * 2.0 * Math.PI;
            const phi = Math.acos(2.0 * v - 1.0);
            
            let r = Math.cbrt(Math.random());
            if (Math.random() > 0.75) r *= 1.05; // Creating an outer fluffy shell overlap
            
            const rx = 8.5;
            const ry = 19.5;
            const rz = 9.5;
            
            x = rx * r * Math.sin(phi) * Math.cos(theta);
            y = ry * r * Math.sin(phi) * Math.sin(theta);
            z = rz * r * Math.cos(phi);
            
            // Vertical tapering (narrow at the top, wider at the base)
            const normalizedY = (y + 19.5) / 39; 
            const taper = 1.0 - (normalizedY * 0.45); 
            x *= taper;
            z *= taper;
            
            // Curve aggressively inwards at the apex (top)
            const curveAmount = (normalizedY * normalizedY) * 3.8;
            x -= xSign * curveAmount;
            
            x += xSign * 11.5;
            
            // Anatomical visual detail: Carving the Cardiac Notch for the left lung
            if (isLeft) {
                // The heart sits mostly left of center, lower quadrant
                if (y > -8 && y < 6 && x > -15 && x < -4 && z > -2 && z < 6) {
                    // Push particles away to carve out space
                    x -= 3;
                    z -= 2;
                    y -= 1;
                }
            }
            
            const tiltAngle = xSign * 0.16;
            const xt = x * Math.cos(tiltAngle) - y * Math.sin(tiltAngle);
            const yt = x * Math.sin(tiltAngle) + y * Math.cos(tiltAngle);
            x = xt;
            y = yt;
            
            // Core complex organic noise giving it a "lobular" internal texture
            const textureNoise = Math.sin(x*1.5)*Math.cos(y*1.5)*Math.sin(z*1.5);
            x += textureNoise * 0.8;
            y += textureNoise * 0.8;
            z += textureNoise * 0.8;
        }
        
        // Micro-jitter for a soft organic feel rather than perfect shapes
        const noiseX = (Math.random() - 0.5) * 1.5;
        const noiseY = (Math.random() - 0.5) * 1.5;
        const noiseZ = (Math.random() - 0.5) * 1.5;
    
        positions[i*3] = x + noiseX;
        positions[i*3+1] = y + noiseY;
        positions[i*3+2] = z + noiseZ;
    
        // Distribute colors to match red branding but add high visual depth
        const distFromCenter = Math.sqrt(x*x + z*z);
        let finalColor = new THREE.Color();
        
        if (randType > 0.92 || Math.random() > 0.98) {
            finalColor.copy(colorWhite); // Bright nerve/bronchi core lines glowing
        } else if (distFromCenter < 6 && Math.random() > 0.4) {
            finalColor.lerpColors(colorWhite, colorRed, 0.65); // Inner glowing layer
        } else {
            // Outer volume, blend natively between dark red and bright red
            finalColor.lerpColors(colorDarkRed, colorRed, Math.random());
            if (Math.random() > 0.85) {
                // Add rich shadow depth randomly to make it look volumetric
                finalColor.lerpColors(finalColor, colorBlack, 0.5);
            }
        }
    
        colors[i*3] = finalColor.r;
        colors[i*3+1] = finalColor.g;
        colors[i*3+2] = finalColor.b;
    
        // Extreme variation in point size for visceral depth
        if (randType > 0.85) {
            sizes[i] = Math.random() * 3.5 + 1.2; // Extra prominent thick bronchi elements
        } else {
            sizes[i] = Math.random() * 2.0 + 0.3; // Normal lung lobe particles
        }
    }
    
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geometry.setAttribute('size', new THREE.BufferAttribute(sizes, 1));
    
    // Custom shader material for glowing round points and organic physics
    const material = new THREE.ShaderMaterial({
        uniforms: {
            time: { value: 0.0 }
        },
        vertexShader: `
            uniform float time;
            attribute float size;
            attribute vec3 color;
            varying vec3 vColor;
            
            void main() {
                vColor = color;
                vec3 pos = position;
                
                // Slower, subtle organic breathing wave
                float breath = sin(time * 2.5) * 0.03 + 1.0;
                float ripple = sin(pos.y * 0.5 - time * 3.0) * 0.02;
                
                pos.x *= breath + ripple;
                pos.z *= breath + ripple;
                pos.y *= (breath * 0.5 + 0.5);
                
                vec4 mvPosition = modelViewMatrix * vec4(pos, 1.0);
                
                // Perspective sizing with distance fade
                gl_PointSize = size * (100.0 / -mvPosition.z) * (1.0 + sin(time * 8.0 + pos.x) * 0.4);
                gl_Position = projectionMatrix * mvPosition;
            }
        `,
        fragmentShader: `
            varying vec3 vColor;
            void main() {
                // Circular, smooth soft particle rendering
                float d = distance(gl_PointCoord, vec2(0.5));
                if (d > 0.5) discard;
                float alpha = smoothstep(0.5, 0.1, d) * 1.0;
                gl_FragColor = vec4(vColor, alpha);
            }
        `,
        transparent: true,
        blending: THREE.NormalBlending,
        depthWrite: false
    });
    
    const lungSystem = new THREE.Points(geometry, material);
    
    // Optional slight tilt to look better in context
    lungSystem.rotation.x = 0.15;
    
    scene.add(lungSystem);
    
    // Animation Loop
    const clock = new THREE.Clock();
    
    function animate() {
        requestAnimationFrame(animate);
        
        const elapsedTime = clock.getElapsedTime();
        
        // Smoothly lerp towards target rotation speed based on hover
        if (lungSystem.userData.targetSpeed !== undefined) {
            lungSystem.userData.currentSpeed += (lungSystem.userData.targetSpeed - lungSystem.userData.currentSpeed) * 0.05;
            lungSystem.rotation.y += lungSystem.userData.currentSpeed;
        } else {
            lungSystem.rotation.y += 0.015;
        }
        
        // Update shader uniforms
        material.uniforms.time.value = elapsedTime;
        
        renderer.render(scene, camera);
    }
    
    animate();
    
    // Handle Window Resize dynamically
    window.addEventListener('resize', () => {
        if (parentContainer) {
            camera.aspect = parentContainer.clientWidth / parentContainer.clientHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(parentContainer.clientWidth, parentContainer.clientHeight);
        }
    });

    // Hover effect pauses/slows the new 3D rotation, just like the old CSS animation
    parentContainer.addEventListener('mouseenter', () => {
        lungSystem.userData.targetSpeed = 0.0008;
    });
    parentContainer.addEventListener('mouseleave', () => {
        lungSystem.userData.targetSpeed = 0.0035;
    });
    
    lungSystem.userData.currentSpeed = 0.0035;
    lungSystem.userData.targetSpeed = 0.0035;
}
