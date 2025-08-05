import jax
from jaxtyping import Float, Array
import jax.numpy as jnp

def bilinear_sample(
    image: Float[Array, "h w c"], 
    x: float,
    y: float
) -> Float[Array, "c"]:
    """
    Bilinear sample an image at floating point coordinates (x, y).
    image: jax array of shape (h, w, c)
    x, y: float coordinates (can be out of bounds, will be clipped)
    Returns: sampled pixel value (c,)
    """
    height, width, _ = image.shape
    x_clamped = jnp.clip(x, 0.0, width - 1.0)
    y_clamped = jnp.clip(y, 0.0, height - 1.0)
    x0 = jnp.floor(x_clamped).astype(jnp.int32)
    x1 = jnp.minimum(x0 + 1, width - 1)
    y0 = jnp.floor(y_clamped).astype(jnp.int32)
    y1 = jnp.minimum(y0 + 1, height - 1)
    x_weight = x_clamped - x0
    y_weight = y_clamped - y0
    top_left_pixel = image[y0, x0]
    top_right_pixel = image[y0, x1]
    bottom_left_pixel = image[y1, x0]
    bottom_right_pixel = image[y1, x1]
    return (
        top_left_pixel * (1 - x_weight) * (1 - y_weight) +
        top_right_pixel * x_weight * (1 - y_weight) +
        bottom_left_pixel * (1 - x_weight) * y_weight +
        bottom_right_pixel * x_weight * y_weight
    )

def match_residual(
    match: Float[Array, "2"],
    target_frame_ray: Float[Array, "3"],
    reference_frame_rays: Float[Array, "h w 3"], 
):
    predicted_reference_ray = bilinear_sample(reference_frame_rays, match[0], match[1])
    residual = predicted_reference_ray - target_frame_ray
    return residual

def match(
    target_frame_rays: Float[Array, "h w 3"],
    reference_frame_rays: Float[Array, "h w 3"],
    max_iters: int = 10,
):
    h, w, _ = reference_frame_rays.shape
    y_coords, x_coords = jnp.meshgrid(jnp.arange(h), jnp.arange(w), indexing='ij')
    matches = jnp.stack([x_coords, y_coords], axis=-1).astype(jnp.float32)
    def gauss_newton_step(matches, _):
        def per_pixel(match, target_ray, ref_rays):
            jac = jax.jacobian(match_residual, argnums=0)(match, target_ray, ref_rays)
            residual = match_residual(match, target_ray, ref_rays)
            damping = 1e-6 * jnp.eye(jac.shape[1], dtype=jac.dtype)
            delta = jnp.linalg.solve(jac.T @ jac + damping, jac.T @ residual)
            return match - delta
        per_pixel_vmap = jax.vmap(jax.vmap(per_pixel, in_axes=(0,0,None)), in_axes=(0,0,None))
        new_matches = per_pixel_vmap(matches, target_frame_rays, reference_frame_rays)
        return new_matches, None
    matches, _ = jax.lax.scan(gauss_newton_step, matches, None, length=max_iters)
    return matches