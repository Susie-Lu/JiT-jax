import logging
import jax
import orbax.checkpoint as ocp

log = logging.getLogger(__name__)


def get_global_array(state_from_pmap):
    """
    Saves a pmap-replicated state in a multi-host environment.

    This function converts the host-local arrays from pmap into
    global jax.Arrays that Orbax can safely serialize.
    """
    log.info(f"Host {jax.process_index()}: Starting checkpoint save...")

    def host_local_to_global(arr):
        """Converts one host-local array to a Global Array."""
        if not isinstance(arr, jax.Array):
            # Don't convert non-arrays (like Python integers for 'step')
            return arr

        # Handles the "de-scrambling" of pmap's device order
        # and stitches the local arrays into one GDA.
        return ocp.utils.fully_replicated_host_local_array_to_global_array(arr)

    log.info(
        f"Host {jax.process_index()}: Converting host-local state to global state..."
    )
    global_state = jax.tree.map(host_local_to_global, state_from_pmap)
    log.info(f"Host {jax.process_index()}: Conversion complete.")

    return global_state
