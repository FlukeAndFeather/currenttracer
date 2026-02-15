# Current Tracer

Goal: Create a website where someone can click on a point on the world's oceans and watch how the point will drift with surface currents.

Experience: A globe showing the world's oceans. When someone clicks on a spot, a glowing point appears. Time advances and the currents carry the glowing point around. A trail forms behind the point, showing its trajectory. The trail is brightest at the point, then fades until it reaches a greyish background. Time can be sped up or slowed down.

Data: Surface currents derived from the Global Ocean Physics Analysis and Forecast https://doi.org/10.48670/moi-00016.

## Python Implementation

### Stack

- **Backend**: FastAPI serving a REST API and static files
- **Frontend**: A single HTML page using CesiumJS for the 3D globe, communicating with the backend via fetch calls
- **Data handling**: xarray + netCDF4 to load and query the Copernicus ocean current fields (eastward `uo` and northward `vo` velocity components)
- **Particle advection**: NumPy/SciPy on the server side, using `scipy.interpolate.RegularGridInterpolator` to bilinearly interpolate velocities at arbitrary lon/lat positions, then a simple RK4 (Runge-Kutta 4th order) time-stepping loop to move the particle

### How it works

1. **Data prep** — Download the Global Ocean Physics Analysis and Forecast NetCDF files covering the desired time range. On startup, the server loads the `uo` and `vo` grids into memory with xarray and builds interpolators for each time step (or lazily on demand via Dask).

2. **Click event** — The user clicks the globe. The frontend sends `POST /trace` with `{ lon, lat, duration_days, dt_hours }`.

3. **Server-side advection** — The endpoint runs an RK4 loop:
   ```python
   pos = np.array([lon, lat])
   trajectory = [pos.copy()]
   for _ in range(n_steps):
       k1 = velocity_at(pos, t)
       k2 = velocity_at(pos + 0.5 * dt * k1, t + 0.5 * dt)
       k3 = velocity_at(pos + 0.5 * dt * k2, t + 0.5 * dt)
       k4 = velocity_at(pos + dt * k3, t + dt)
       pos += (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
       t += dt
       trajectory.append(pos.copy())
   ```
   `velocity_at` converts m/s velocities to degrees/s (accounting for `cos(lat)` for the zonal component) and interpolates from the gridded data.

4. **Response** — The server returns the trajectory as a JSON array of `[lon, lat, time]` tuples.

5. **Rendering** — The frontend draws the trajectory on the globe as a polyline with a color/opacity gradient: bright at the particle head, fading to grey along the tail. A playback slider lets the user control animation speed.

### Project structure

```
currenttracer/
  server.py          # FastAPI app, /trace endpoint, advection logic
  data.py            # xarray loading, interpolator construction
  static/
    index.html       # Globe, click handler, trajectory rendering
    style.css
  data/              # Downloaded NetCDF files (gitignored)
  requirements.txt   # fastapi, uvicorn, xarray, netcdf4, scipy, numpy
```

### Deployment

**Hosting**: DigitalOcean droplet (flat-rate VPS, no surprise billing)
- Start with a $12/mo droplet (2 GB RAM, 1 vCPU) — enough to hold the current data in memory and handle moderate traffic
- Scale up to a larger droplet later if needed

**Setup**:
1. Create an Ubuntu droplet on DigitalOcean
2. SSH in, install Python 3.11+, clone the repo, install dependencies
3. Download the NetCDF data files to `data/`
4. Set up uvicorn as a systemd service (auto-restarts on crash/reboot)
5. Install Caddy as a reverse proxy — handles HTTPS automatically via Let's Encrypt
6. Point a domain name at the droplet's IP address

**Security**:
- **Cloudflare** (free tier) in front of the domain — absorbs basic DOS attacks, hides the server's real IP
- **Rate limiting** — `slowapi` middleware on the `/trace` endpoint (e.g. 5 requests/min per IP)
- **Input bounds** — Server-side caps on `duration_days` (max 365) and minimum `dt_hours` (min 1) to prevent expensive computations
- **Computation timeout** — Hard timeout on the advection loop so no single request can hang a worker