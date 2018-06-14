using CNTK;

namespace SiaNet.Model.Layers
{
    using System.Dynamic;

    /// <summary>
    /// Reshapes an output to a certain shape.
    /// </summary>
    public class Splice : LayerConfig
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="Splice"/> class.
        /// </summary>
        internal Splice()
        {
            base.Name = "Splice";
            base.Params = new ExpandoObject();
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="Splice"/> class.
        /// </summary>
        /// <param name="appendingVector">The target shape of the output.</param>
        public Splice(Axis axis, VariableVector appendingVector)
            : this()
        {
            AppendingVector = appendingVector;
            Axis = axis;
        }


        /// <summary>
        /// List of integers. Does not include the batch axis.
        /// </summary>
        /// <value>
        /// The target shape.
        /// </value>
        [Newtonsoft.Json.JsonIgnore]
        public VariableVector AppendingVector
        {
            get
            {
                return base.Params.AppendingVector;
            }

            set
            {
                base.Params.AppendingVector = value;
            }
        }

        public Axis Axis
        {
            get { return base.Params.Axis; }
            set { base.Params.Axis = value; }
        }
    }
}
